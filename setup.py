import configparser
import os
import os.path as op
import platform
import subprocess
import sys
from distutils.core import setup
from distutils.sysconfig import get_python_lib
from shutil import copyfile

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install

# --- Current version

VERSION = "0.3.0"

# --- Read information from main package

config = configparser.ConfigParser()
config.read('cpp/config.ini')
informations = config["informations"]
author = config["author"]
version = informations["version"]
with open("README.md", "r") as fh:
    long_description = fh.read()


# --- Custom build classes

class VerifyVersionCommand(install):
    """Custom command to verify that the git tag matches our version"""
    description = 'verify that the git tag matches our version'

    def run(self):
        global version

        tag = os.getenv('CIRCLE_TAG')

        if tag != "v%s" % version:
            info = "Git tag: {0} does not match the version of this app: {1}".format(
                tag, version
            )
            sys.exit(info)


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: " +
                ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)',
                                         out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath(ext.name)))

        extdir = op.join(extdir, "experimaestro")
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(
                cfg.upper(),
                extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j2']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get('CXXFLAGS', ''),
            self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        
        # Copy api.h
        copyfile('./cpp/include/public/xpm/api.h', op.join(extdir, "api.h"))

        # Configuring
        print(['cmake', ext.sourcedir] + cmake_args)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args,
                              cwd=self.build_temp, env=env)
                              
        # Building
        subprocess.check_call(['cmake', '--build', '.', '--target', 'experimaestro_shared'] + build_args,
                              cwd=self.build_temp)

        print()  # Add an empty line for cleaner output

# --- Setup

setup(
    # Basic informatoin
    name='experimaestro',
    version=version,
    description=long_description,
    long_description_content_type="text/markdown",
    author=author["name"],
    author_email=author["email"],
    url=informations["url"],

    # Packages
    packages=['experimaestro'],
    package_dir = {'experimaestro': 'experimaestro'},

    # Allows to use a custom build step
    ext_modules=[CMakeExtension('experimaestro', 'cpp')],

    # We do not allow archives
    zip_safe=False,

    # Version verification
    cmdclass={
        'verify': VerifyVersionCommand,
        'build_ext': CMakeBuild
    }
)
