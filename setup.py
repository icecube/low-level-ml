#!/usr/bin/env python
import os
from setuptools import setup, find_packages

here = os.path.dirname(__file__)

about = {}
with open(os.path.join(here, 'low_level_ml', '__about__.py')) as fobj:
    exec(fobj.read(), about)

setup(
    name='low_level_ml',
    version=about['__version__'],
    packages=find_packages(),
    install_requires=[
        'numpy', 'scipy', 'matplotlib', 'pandas', 'sklearn',
    ],
    include_package_data=True,
    author=about['__author__'],
    author_email=about['__author_email__'],
    maintainer=about['__author__'],
    maintainer_email=about['__author_email__'],
    description=about['__description__'],
    url=about['__url__']
)
