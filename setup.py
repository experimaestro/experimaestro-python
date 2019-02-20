import configparser
import logging
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

logging.basicConfig(level=logging.DEBUG)

with open("README.md", "r") as fh:
    long_description = fh.read()


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

config = configparser.ConfigParser()
config.read('cpp/config.ini')
informations = config["informations"]
author = config["author"]

def experimaestro_test_suite():
    import unittest
    print(sys.path)
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests', pattern='test_*.py')
    return test_suite

setup(name='experimaestro',
    version=informations["version"],
    description=long_description,
    long_description_content_type="text/markdown",
    author=author["name"],
    author_email=author["email"],
    url=informations["url"],
    packages=['experimaestro'],
    package_dir = {'experimaestro': 'experimaestro'},
    # add custom build_ext command
    # test_suite="setup.experimaestro_test_suite",
    cmdclass=dict(build_ext=CMakeBuild),
    ext_modules=[CMakeExtension('experimaestro', 'cpp')],
    zip_safe=False,
)
