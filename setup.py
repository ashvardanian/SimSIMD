import os
import sys
import re
import platform
from os.path import dirname
import multiprocessing
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

__version__ = open('VERSION', 'r').read().strip()
__lib_name__ = 'simsimd'


this_directory = os.path.abspath(dirname(__file__))
with open(os.path.join(this_directory, 'README.md')) as f:
    long_description = f.read()


class CMakeExtension(Extension):
    def __init__(self, name, source_dir=''):
        Extension.__init__(self, name, sources=[])
        self.source_dir = os.path.abspath(source_dir)


class CMakeBuild(build_ext):
    def build_extension(self, ext):

        self.parallel = multiprocessing.cpu_count() // 2
        extension_dir = os.path.abspath(dirname(
            self.get_ext_fullpath(ext.name)))

        # required for auto-detection & inclusion of auxiliary 'native' libs
        if not extension_dir.endswith(os.path.sep):
            extension_dir += os.path.sep

        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extension_dir}',
            f'-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY={extension_dir}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
        ]

        if sys.platform.startswith('darwin'):
            archs = re.findall(r'-arch (\S+)', os.environ.get('ARCHFLAGS', ''))
            if archs:
                cmake_args += [
                    '-DCMAKE_OSX_ARCHITECTURES={}'.format(';'.join(archs))]

        build_args = []
        if sys.platform.startswith('win32'):
            build_args += ['--config', 'Release']

        subprocess.check_call(['cmake', ext.source_dir] + cmake_args)
        subprocess.check_call(
            ['cmake', '--build', '.', '--target', "py_simsimd"] + build_args)

    def run(self):
        build_ext.run(self)


setup(
    name=__lib_name__,
    version=__version__,

    author='Ashot Vardanian',
    author_email='info@unum.cloud',
    url='https://github.com/ashvardanian/simsimd',
    description='CPython wrapper to connect with USearch',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='Apache-2.0',

    classifiers=[
        'Development Status :: 5 - Production/Stable',

        'Natural Language :: English',
        'Intended Audience :: Developers',
        'Intended Audience :: Information Technology',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: Linux',
        'Programming Language :: C',
        'Programming Language :: Python :: Implementation :: CPython',
        'Topic :: Communications :: File Sharing',
        'Topic :: Internet :: WWW/HTTP',
        'Topic :: System :: Networking',
    ],

    ext_modules=[
        CMakeExtension('simsimd'),
    ],
    cmdclass={
        'build_ext': CMakeBuild,
    },
    zip_safe=False,
    python_requires='>=3.9',
)
