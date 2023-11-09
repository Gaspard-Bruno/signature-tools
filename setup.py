import importlib.util
import os
import sys

from setuptools import find_packages, setup

DISTNAME = "signature"
DESCRIPTION = "Computer Vision and Image processing library"
PYTHON_VERSION = ">=3.10"
AUTHOR = "Marco João"
KEYWORDS = "machine learning, computer vision, image processing, ai, image generation, ml"
VERSION = "0.0.0"
with open("src/signature/version.py", encoding="utf-8") as fid:
    for line in fid:
        if line.startswith("__version__"):
            VERSION = line.strip().split()[-1][1:-1]
            break

with open("README.md", encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()


def __find_in_path(name, path):
    for directory in path.split(os.pathsep):
        binpath = os.path.join(directory, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None


def __is_cuda_available() -> bool:
    if "CUDAHOME" in os.environ:
        home = os.environ["CUDAHOME"]
        nvcc = os.path.join(home, "bin", "nvcc")
    else:
        # otherwise, search the PATH for NVCC
        nvcc = __find_in_path("nvcc", os.environ["PATH"])
        if nvcc is None:
            return False
        home = os.path.dirname(os.path.dirname(nvcc))
    cudaconfig = {
        "home": home,
        "nvcc": nvcc,
        "include": os.path.join(home, "include"),
        "lib64": os.path.join(home, "lib64"),
    }

    for _, value in cudaconfig.items():
        if not os.path.exists(value):
            return False
    return cudaconfig is not None


def __get__onnxruntime() -> list[str]:
    name = "onnxruntime"
    if name in sys.modules:
        return []

    if (spec := importlib.util.find_spec(name)) is not None:
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        loader = spec.loader
        if loader is not None:
            loader.exec_module(module)
        return []
    version = "1.16.1"
    return [f"onnxruntime-gpu=={version}" if __is_cuda_available() else f"onnxruntime=={version}"]


def __get_requirements():
    requirements = __get__onnxruntime()

    with open("requirements.txt") as file:
        requirements.extend(file.read().splitlines())
    return requirements


setup(
    name=DISTNAME,
    version=VERSION,
    python_requires=PYTHON_VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    keywords=KEYWORDS,
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=__get_requirements(),
)
