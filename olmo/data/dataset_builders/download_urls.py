import dataclasses
import hashlib
import io
import json
import logging
import multiprocessing
import os
import pickle
import sys
import time
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from os import rename, makedirs
from os.path import join, exists, relpath
from typing import Union, Dict, Optional
from urllib.parse import urlparse

import PIL.Image
import datasets
import httpx
import numpy as np
import requests
import urllib3
from PIL import ImageFile
from olmo.util import flatten_list
from urllib3.util import Retry
from requests.adapters import HTTPAdapter

from tqdm import tqdm

from olmo.data.dataset import DATA_HOME
from olmo.io import _s3_get_bytes_range



def setup_pil():
    PIL.Image.MAX_IMAGE_PIXELS = None
    ImageFile.LOAD_TRUNCATED_IMAGES = True


@dataclasses.dataclass
class DownloadError:
    url: str
    exception: Union[str, Exception]


@dataclasses.dataclass
class ImageError:
    url: str
    exception: Optional[Exception] = None


def compute_hash(string: Union[str, bytes]) -> str:
    if isinstance(string, str):
        return hashlib.sha256(string.encode("utf-8")).hexdigest()
    else:
        return hashlib.sha256(string).hexdigest()


client = None


def init_worker():
    global client
    client = httpx.Client(timeout=httpx.Timeout(20, connect=5), follow_redirects=True)
    setup_pil()


def _validate_image(image_bytes, image_sha, check_sha):
    if check_sha:
        if compute_hash(image_bytes) != image_sha:
            return ValueError("Mismatched image hash")
    else:
        try:
            with warnings.catch_warnings(record=True):
                img = PIL.Image.open(io.BytesIO(image_bytes))
                if min(img.size) == 0:
                    raise ValueError("Zero dimensional image")
                img.verify()
        except Exception as exc:
            return exc
    return None


def _download_images(args):
    url, image_sha, check_sha, cache_only, src_dir, kwargs = args
    global client
    internal_url = None
    if isinstance(image_sha, tuple):
        image_sha, internal_url = image_sha
    image_id = compute_hash(url)
    cache_file = join(src_dir, image_id)

    if exists(cache_file):
        try:
            with open(cache_file, "rb") as stream:
                cached_bytes = stream.read()
            if _validate_image(cached_bytes, image_sha, check_sha) is None:
                return url, cache_file
        except OSError:
            pass
        # An old attempt may have cached an exception string instead of an image.
        # Keep that file until a valid replacement is ready, then atomically replace it.
    if cache_only:
        return DownloadError(url, "Missing or invalid cached image")

    for attempt in range(3):
        try:
            response = client.get(url)
            response.raise_for_status()
            image_bytes = response.content
            validation_error = _validate_image(image_bytes, image_sha, check_sha)
            if validation_error is not None:
                return ImageError(url, str(validation_error))
            temp_file = f"{cache_file}.{os.getpid()}.tmp"
            with open(temp_file, "wb") as stream:
                stream.write(image_bytes)
            os.replace(temp_file, cache_file)
            return url, cache_file
        except httpx.HTTPStatusError as exc:
            reason = f"HTTP {exc.response.status_code}"
            if exc.response.status_code in (400, 401, 403, 404, 410, 451):
                return DownloadError(url, reason)
        except httpx.TooManyRedirects as exc:
            return DownloadError(url, f"TooManyRedirects: {exc}")
        except (httpx.RequestError, OSError, ValueError) as exc:
            reason = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, httpx.ConnectError) and "Network is unreachable" in str(exc):
                return DownloadError(url, reason)
        if attempt < 2:
            time.sleep(2 ** attempt)
    return DownloadError(url, reason)


def download_pixmo_urls(
    data: datasets.Dataset,
    n_processes,
    check_sha,
    output_dir,
    request_kwargs=None,
    cache_only=False,
    verify=True
) -> Dict[str, str]:
    """Download urls from a PixMo dataset, return a map of urls->filename"""
    if "image_urls" in data.features:
        assert not check_sha
        urls = set(flatten_list(data["image_urls"]))
        urls_and_shas = [(url, None) for url in urls]
    elif check_sha:
        urls_and_shas = list(dict(zip(data["image_url"], data["image_sha256"])).items())
    else:
        urls_and_shas = [(url, None) for url in list(set(data["image_url"]))]

    # Randomize order so resuming is more convenient, speed is more predictable,
    # and to distribute requests across different domains
    urls_and_shas.sort(key=lambda x: x[0])
    np.random.RandomState(58713).shuffle(urls_and_shas)

    logging.info(f"Getting files for {len(urls_and_shas)} image URLs")
    makedirs(output_dir, exist_ok=True)
    if request_kwargs is None:
        request_kwargs = dict(timeout=60)
    if not verify:
        request_kwargs["verify"] = False
        urllib3.disable_warnings()

    images = []
    to_save = [(url, image_sha, check_sha, cache_only, output_dir, request_kwargs) for url, image_sha in urls_and_shas]
    pbar = tqdm(total=len(to_save), desc=f"{0}/{len(to_save)}", disable=not sys.stderr.isatty())
    image_error, download_err, success = 0, 0, 0
    errors_by_reason = Counter()
    errors_by_host = Counter()
    failure_manifest = os.environ.get("PIXMO_FAILURE_MANIFEST")
    failure_stream = open(failure_manifest, "w", encoding="utf-8") if failure_manifest else None
    if failure_stream:
        failure_stream.write("url\ttype\terror\n")

    if n_processes != 1:
        def _iter():
            with multiprocessing.Pool(processes=n_processes, initializer=init_worker) as pool:
                for val in pool.imap_unordered(_download_images, to_save):
                    yield val
    else:
        init_worker()
        def _iter():
            for val in to_save:
                yield _download_images(val)

    found_urls = {}
    try:
        for index, val in enumerate(_iter(), 1):
            if isinstance(val, (ImageError, DownloadError)):
                if isinstance(val, ImageError):
                    image_error += 1
                else:
                    download_err += 1
                reason = str(val.exception).replace("\t", " ").replace("\n", " ")
                errors_by_reason[reason.split(":", 1)[0]] += 1
                errors_by_host[urlparse(val.url).hostname] += 1
                if failure_stream:
                    failure_stream.write(f"{val.url}\t{type(val).__name__}\t{reason}\n")
            else:
                url, filename = val
                found_urls[url] = filename
                success += 1
            pbar.update(1)
            if index % 5000 == 0:
                logging.info(
                    "PixMo URLs %d/%d: success=%d download_errors=%d image_errors=%d; top reasons=%s; top hosts=%s",
                    index, len(to_save), success, download_err, image_error,
                    errors_by_reason.most_common(5), errors_by_host.most_common(5),
                )
                if failure_stream:
                    failure_stream.flush()
    finally:
        pbar.close()
        if failure_stream:
            failure_stream.close()
    logging.info(f"Got images for {len(found_urls)}/{len(urls_and_shas)} ({len(found_urls)/len(urls_and_shas)*100:0.2f}%) image URLs")
    return found_urls


def filter_and_group_data(data: datasets.Dataset, url_to_path: Dict, check_sha: bool) -> datasets.Dataset:
    """
    Groups a pixmo datasets so each row contains all annotation for one image, and add
    images path using `url_to_path`, removing rows that do not exist in `url_to_path`
    """
    grouped_by_url = defaultdict(list)
    for example in data:
        if example["image_url"] not in url_to_path:
            continue
        grouped_by_url[example["image_url"]].append(example)

    grouped_examples = []
    for image_url, examples in grouped_by_url.items():
        grouped = dict(
            image_url=image_url,
            image=url_to_path[image_url],
        )
        if "image_sha256" in examples[0] and not check_sha:
            assert all(examples[0]["image_sha256"] == ex["image_sha256"] for ex in examples)
            grouped["original_sha256"] = examples[0]["image_sha256"]
        annotations = defaultdict(list)
        for ex in examples:
            for k, v in ex.items():
                if k not in ["image_url", "image_sha256"]:
                    annotations[k].append(v)
        grouped.update(annotations)
        grouped_examples.append(grouped)
    return datasets.Dataset.from_list(grouped_examples)
