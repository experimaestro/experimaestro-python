import configparser
import os
import os.path as op
import platform
import subprocess
import sys
from distutils.core import setup
from distutils.sysconfig import get_python_lib
from shutil import copyfile, copytree, rmtree

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install

# --- Current version

VERSION = "0.4.0"

# --- Read information from main package

config = configparser.ConfigParser()
config.read('config.ini')
informations = config["informations"]
description = informations["description"]
author = config["author"]
cppversion = informations["version"]
with open("README.md", "r") as fh:
    long_description = fh.read()


# --- Custom build classes

class VerifyVersionCommand(install):
    """Custom command to verify that the git tag matches our version"""
    description = 'verify that the git tag matches our version'

    def run(self):
        global VERSION

        tag = os.getenv('CIRCLE_TAG')

        if tag != "v%s" % VERSION:
            info = "Git tag: {0} does not match the version of this app: {1}".format(
                tag, VERSION
            )
            sys.exit(info)

# --- Setup

setup(
    # Basic informatoin
    name='experimaestro',
    version=VERSION,
    description=description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=author["name"],
    author_email=author["email"],
    url=informations["url"],

    # Packages
    packages=['experimaestro'],
    package_dir = {'experimaestro': 'experimaestro'},

    # We do not allow archives
    zip_safe=False,

    # Use MANIFEST.in
    include_package_data=True,

    # Version verification
    cmdclass={
        'verify': VerifyVersionCommand
    }
)
