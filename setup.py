from distutils.core import setup

setup(
    name="sinabs",
    version="0.1dev",
    packages=["sinabs"],
    license="All rights reserved aiCTX AG",
    install_requires=['numpy', 'pandas', 'tensorflow', 'torch'],
)
