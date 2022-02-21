# Contributing to sinabs

We welcome developers to build and contribute to sinabs.

Please email sinabs@synsense.ai for a contributors license agreement. 


## How do you go about it?

Short answer: fork the repository using your own account, make changes and commit them in a new branch and finally open a pull request on Github.

A more detailed explanation can be found [here](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests).

## Coding style

Please format your code before opening a pull request. We use [black](https://black.readthedocs.io/en/stable/index.html) code style. 

## Testing

Please add tests for any new features you're contributing. We use `pytest` for testing the library. 
Install the necessary packages by running the following command.

```
$ pip install -r tests/requirements.txt
```

All tests are located in the `tests/` folder and can be run using `pytest`.

```
$ cd /path/to/sinabs/
$ pytest
```

It is critical that your additions have a corresponding test case and *all* current tests pass for a merge request be accepted.