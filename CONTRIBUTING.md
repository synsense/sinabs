# Contributing to sinabs

We welcome developers to build and contribute to sinabs.

Please email sinabs@synsense.ai for a contributors license agreement. 


## How do you go about it?

Short answer: fork, make changes, merge request to sinabs.

**Develop from**: `master` branch of sinabs.


The most straight-forward workflow to contribute would be to fork the repository and make your changes.
Once you finalize your changes, please create a merge request.

Please see gitlab's explanation on [Forking Workflow](https://docs.gitlab.com/ee/workflow/forking_workflow.html) 
for a detailed explanation.

## Coding style

Please adhere to the coding style of the library when you develop your contributions.
We use pep8 + black code style and formatting. 

## Testing

We use `pytest` for testing the library. 
Install the necessary packages by running the following command.

```
$ pip install -r test-requirements.txt
```

All tests are located in the `tests/` folder and can be run using `pytest`.

```
$ cd /path/to/sinabs/
$ pytest
```

It is critical that your additions have a corresponding test case and *all* current tests pass for a merge request be accepted.