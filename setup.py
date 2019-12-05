#  Copyright (c) 2019-2019     aiCTX AG (Sadique Sheik, Qian Liu).
#
#  This file is part of sinabs
#
#  sinabs is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  sinabs is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with sinabs.  If not, see <https://www.gnu.org/licenses/>.

from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="sinabs",
    author="aiCTX AG",
    author_email="sadique.sheik@aictx.ai",
    version="0.1.dev5",
    description="A spiking deep neural network simulator, and neuromoprhic hardware emulator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=["sinabs", "sinabs.layers", "sinabs.from_keras"],
    license="GNU AGPLv3, Copyright (c) 2019 aiCTX AG",
    install_requires=["numpy", "pandas", "torch"],
    python_requires=">=3.6",
    project_urls={
        "Source": "https://gitlab.com/aiCTX/sinabs/",
        "Documentation": "https://aictx.gitlab.io/sinabs",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
    ],
)
