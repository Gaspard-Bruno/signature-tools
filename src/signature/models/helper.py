import os
import sys

from urllib.parse import urlparse
import torch
from ..logger import console as logger
from torch.hub import download_url_to_file, get_dir
import hashlib

def sha256_check(filename) -> str:

    with open(filename,"rb") as f:
        bytes = f.read() # read entire file as bytes
        readable_hash = hashlib.sha256(bytes).hexdigest();
        return readable_hash


def switch_mps_device(model_name, device):
    if str(device) == "mps":
        logger.log(f"{model_name} not support mps, switch to cpu")
        return torch.device("cpu")
    return device


def get_cache_path_by_url(url):
    parts = urlparse(url)
    hub_dir = get_dir()
    model_dir = os.path.join(hub_dir, "checkpoints")
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    filename = os.path.basename(parts.path)
    cached_file = os.path.join(model_dir, filename)
    return cached_file


def download_model(url, model_sha256: str):
    cached_file = get_cache_path_by_url(url)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        hash_prefix = None
        download_url_to_file(url, cached_file, hash_prefix, progress=True)
        if model_sha256:
            _sha256 = sha256_check(cached_file)
            if model_sha256 == _sha256:
                logger.log(f"Download model success, sha256: {_sha256}")
            else:
                try:
                    os.remove(cached_file)
                    logger.log(
                        f"Model sha256: {_sha256}, expected sha256: {model_sha256}, wrong model deleted."
                    )
                except:
                    logger.log(
                        f"Model sha256: {_sha256}, expected sha256: {model_sha256}, please delete {cached_file}."
                    )
                exit(-1)

    return cached_file



def handle_error(model_path, model_sha256, e):
    _sha256 = sha256_check(model_path)
    if _sha256 != model_sha256:
        try:
            os.remove(model_path)
            logger.log(
                f"Model sha256: {_sha256}, expected sha256: {model_sha256}, wrong model deleted."
            )
        except:
            logger.log(
                f"Model sha256: {_sha256}, expected sha256: {model_sha256}, please delete {model_path}."
            )
    else:
        logger.log(
            f"Failed to load model {model_path}"
        )
    exit(-1)


def load_jit_model(url_or_path, device, model_sha256: str):
    if os.path.exists(url_or_path):
        model_path = url_or_path
    else:
        model_path = download_model(url_or_path, model_sha256)

    logger.log(f"Loading model from: {model_path}")
    try:
        model = torch.jit.load(model_path, map_location="cpu").to(device) # type: ignore
    except Exception as e:
        handle_error(model_path, model_sha256, e)
    model.eval() # type: ignore
    return model # type: ignore


def load_model(model: torch.nn.Module, url_or_path, device, model_sha256):
    if os.path.exists(url_or_path):
        model_path = url_or_path
    else:
        model_path = download_model(url_or_path, model_sha256)

    try:
        logger.log(f"Loading model from: {model_path}")
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
    except Exception as e:
        handle_error(model_path, model_sha256, e)
    model.eval()
    return model