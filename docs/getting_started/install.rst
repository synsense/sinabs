Installation
------------

**Sinabs** is available on PyPI::

    pip install sinabs

Checkout our quick instructional on how to create a project based on **sinabs** within a virtual environment using :doc:`pyenv+pipenv<./python_pyenv_pipenv>`.

You can also install the latest/nightly build of sinabs by installing the pre-releases.::

    pip install --upgrade sinabs --pre

Alternatively you can also use conda::

    conda install sinabs -c conda-forge

If you want to develop or have access to source code of **sinabs**, download the package from the git repository::

    cd <to/your/software/folder>
    git clone git@github.com:synsense/sinabs.git
    cd sinabs
    pip install .


For developers, we recommend that you install this package as a development version so that you can update the package without reinstalling it.

    pip install -e .

.. note:: 
    If you are either using or developing plugin packages such as `sinabs-exodus`, a development install with `-e` flag does not work as expected. 
    We suggest you do a regular pip installation every time you make a change.
 
 