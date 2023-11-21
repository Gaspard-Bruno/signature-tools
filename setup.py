try:
    from setuptools import find_packages, setup
except ImportError:
    import subprocess
    subprocess.check_call(['pip3', 'install', 'setuptools'])
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

def __get_requirements():
    requirements = []
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
