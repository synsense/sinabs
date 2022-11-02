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

## Releasing
Releasing a new version of Sinabs is automated using [Github actions](https://github.com/synsense/sinabs/actions) and [pbr](https://docs.openstack.org/pbr/latest/). When a new commit is pushed on the main branch, a new build will be pushed to [PYPI](https://pypi.org/project/sinabs/#history) based on the latest tag available, provided that all tests have passed. A tag is always associated with a commit (hash) and therefore persists across branches (which also point to commits). If the latest commit is tagged with something like 'v1.2.3' then this will be the release version name. If the release pipeline is started without the release tag being available, the latest commit will not have a tag. pbr then names this something like 'v1.2.4.dev1', to indicate a development release. Overall, if you're a co-maintainer of Sinabs, please follow these steps to create a new release:

1. Merge/push all your changes into the develop branch.
2. Make sure the tests are passing on the develop branch.
3. Push commits on the develop branch.
4. Tag the commit. Using `git tag` you can check the previous tags. Increment your version accordingly using e.g. `git tag v1.2.4`. We try to stick to [semantic versioning](https://semver.org/). If you botched the tag name, you can delete/retag before you then push the tags with `git push --tags`. 
5. Merge develop into main and push main.

Once the tests passed on Github actions, a new release will be built and pushed to [PYPI](https://pypi.org/project/sinabs/#history)!
