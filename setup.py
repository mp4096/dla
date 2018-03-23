"""Setup file for DLA."""

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name="dla",
    version="0.1",
    description="A simple toolbox for prototyping "
    "distributed linear algebra algorithms.",
    license="BSD",
    author="Mikhail Pak <mikhail.pak@tum.de>",
    packages=["dla"],
    install_requires=[
        "bokeh",
        "mesa",
        "networkx",
        "numpy",
        "scipy",
        ]
    )
