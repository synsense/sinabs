from distutils.core import setup

setup(
    name="sinabs-dynapcnn",
    version="0.0.1dev",
    packages=["sinabs.backend.dynapcnn"],
    license="All rights reserved aiCTX AG",
    install_requires=["sinabs", "samna>=0.3"],
)
