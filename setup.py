from pathlib import Path
from setuptools import setup

basepath = Path(__file__).parent

setup(
    install_requires=(basepath / "requirements.txt").read_text(),
)
