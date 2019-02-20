import os.path as op
import sys
import os
import logging
import platform
import subprocess


from distutils.core import setup
from distutils.command.build import build as _build
from distutils.command.install import install as _install
from distutils.sysconfig import get_python_lib

from setuptools.command.build_ext import build_ext
from setuptools import setup, Extension


logging.basicConfig(level=logging.DEBUG)
# Read the configuration
import configparser


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
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args,
                              cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.', '--target', 'experimaestro_shared'] + build_args,
                              cwd=self.build_temp)
        print()  # Add an empty line for cleaner output

config = configparser.ConfigParser()
config.read('cpp/config.ini')
informations = config["informations"]
author = config["author"]


setup(name='experimaestro',
      version=informations["version"],
      description=informations["description"],
      author=author["name"],
      author_email=author["email"],
      url=informations["url"],
      packages=['experimaestro'],
      package_dir = {'experimaestro': 'experimaestro'},
      data_files=[
          ('.', ['./cpp/include/public/xpm/api.h'])
      ],
      # add custom build_ext command
      cmdclass=dict(build_ext=CMakeBuild),
      ext_modules=[CMakeExtension('experimaestro', 'cpp')],
      zip_safe=False,
)
