# Contributing Guidelines

Thank you for your interest in contributing to our project. Whether it's a new NKI kernel, improving existing kernel code, bug fix, new feature, correction, or additional
documentation, we greatly value feedback and contributions from our community.

Please read through this document before submitting any issues or pull requests to ensure we have all the necessary
information to effectively respond to your bug report or contribution.


## Reporting Bugs/Feature Requests

We welcome you to use the GitHub issue tracker to report bugs or suggest features.

When filing an issue, please check [existing open](https://github.com/aws-neuron/nki-samples/issues), or [recently closed](https://github.com/aws-neuron/nki-samples/issues?utf8=%E2%9C%93&q=is%3Aissue%20is%3Aclosed%20), issues to make sure somebody else hasn't already
reported the issue. Please try to include as much information as you can. Details like these are incredibly useful:

* A reproducible test case or series of steps
* The version of our code being used
* Any modifications you've made relevant to the bug
* Anything unusual about your environment or deployment


## Contributing via Pull Requests
Contributions via pull requests are much appreciated. Before sending us a pull request, please ensure that:

1. You are working against the latest source on the *main* branch.
2. You check existing open, and recently merged pull requests to make sure someone else hasn't addressed the problem already.

To send us a pull request, please:

1. Fork the repository.
2. Modify the source; please focus on the specific changes you are contributing. If you also reformat all the code, it will be hard for us to focus on your change.
3. Please ensure your change satisfies the requirements listed in [Testing Requirements](#testing-requirements) and [Coding Guidelines](#coding-guidelines)
4. Commit to your fork using clear commit messages.
5. Send us a pull request, answering any default questions in the pull request interface.
6. Wait for a repository collaborator to look at your pull request, run the automated tests, and review. If additional changes or discussion is needed, a collaborator will get back to you, so please stay involved in the conversation.
    * Note: pull requests will be tested, staged, and released in a process internal to the Neuron team. Changes will be reflected in a subsequent release

GitHub provides additional document on [forking a repository](https://help.github.com/articles/fork-a-repo/) and
[creating a pull request](https://help.github.com/articles/creating-a-pull-request/).

### Testing Requirements
Running the binaries for a NKI kernel require Neuron devices on an AWS EC2 instance from trn1, trn1n, or inf2 instance families. 
Details on setting up an instance can be found in [here](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-setup.html).

If you would like to test your kernel without requiring a Neuron device, you can use `nki.simulate()` to run your kernel using `NumPy` input/output tensors and types. 
An example can be found in the [layernorm tutorial test](test/unit/test_tutorials_layernorm.py). However, kernels with _only_ simulation tests will not be accepted.

#### Requirements for Kernels Targeting `src/reference/`

All kernels located in this folder need to have the following tests.

1. Numeric accuracy tests with `nki.baremetal`. The output from the kernel
must be validated against a CPU reference implementation. See `test_flash_attn_fwd_numerical` in [test_flash_attn_fwd.py](test/unit/test_flash_attn_fwd.py) as an example. Documentation for `nki.baremetal` is available at [here](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/api/generated/nki.baremetal.html).

2. Performance benchmark tests with `nki.benchmark`. The unit test must have performance checks. At a minimum, put an assertion to verify p99 latency meets a certain threshold. See `test_flash_attn_fwd_perf` in [test_flash_attn_fwd.py](test/unit/test_flash_attn_fwd.py) as an example. Documentation for `nki.benchmark` is available at [here](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/api/generated/nki.benchmark.html)

3. End-to-End integration tests that use your kernel in a model. 
    
    a. Each test should be in its own separate folder.

    b. Each Test must have a `run.sh` script, that accepts an argument \<path to test_result.json\>. See [run.sh of FlashAttention](test/integration/flash_attention/run.sh) as an example. 

    c. The test scripts must produce benchmark results with the `benchmark` function, located in [LatencyCollector.py](test/integration/perf_utils/LatencyCollector.py). The `benchmark` function will write the latency of your E2E model to the `test_result.json`.

    d. Register your test target in [run_integration.sh](test/integration/run_integration.sh).


### Coding Guidelines
Most guidelines are covered by a **PEP-8** check on all newly submitted code, which covers aspects such as code layout and basic Python naming conventions. 
In addition to PEP-8, we use the following NKI specific style guidelines:

1. **Abbreviations**
    * Importing NKI modules should use consistent names. For example,
    ```
    import neuronxcc.nki as nki
    import neuronxcc.nki.isa as nisa
    import neuronxcc.nki.language as nl
    import neuronxcc.nki.typing as nt
    import numpy as np
    ```   
2. Variable Names
    * Indexing should specify partition and free dimensions along with the variable they are used for. For example:
        The index for the partition dimension for tile `a` would be
        ```
        i_p_a = nl.arange(128)[:, None]
        ```
        while the index for the free dimension for tile `b` would be
        ```
        i_f_b = nl.arange(512)[None, :]
        ```
    * Name loop variables, indices, and buffers consistently, and specify their intended use in the name.

3. Documentation
   * New kernels should containing inline docstrings that describe the semantics of the kernel, and provide information on the IO layout. 
   Upon release, we generate the documentation for our kernels and merge them into the NKI API documentation which will appear in the official AWS NKI documentation. 


## Finding Contributions to Work on
Looking at the existing issues is a great way to find something to contribute on. As our projects, by default, use the default GitHub issue labels (enhancement/bug/duplicate/help wanted/invalid/question/wontfix), looking at any ['help wanted'](https://github.com/aws-neuron/nki-samples/labels/help%20wanted) issues is a great place to start.


## Code of Conduct
This project has adopted the [Amazon Open Source Code of Conduct](https://aws.github.io/code-of-conduct).
For more information see the [Code of Conduct FAQ](https://aws.github.io/code-of-conduct-faq) or contact
opensource-codeofconduct@amazon.com with any additional questions or comments.


## Security Issue Notifications
If you discover a potential security issue in this project we ask that you notify AWS/Amazon Security via our [vulnerability reporting page](http://aws.amazon.com/security/vulnerability-reporting/). Please do **not** create a public github issue.


## Licensing

See the [LICENSE](https://github.com/aws-neuron/nki-samples/blob/main/LICENSE.txt) file for our project's licensing. We will ask you to confirm the licensing of your contribution.

We may ask you to sign a [Contributor License Agreement (CLA)](http://en.wikipedia.org/wiki/Contributor_License_Agreement) for larger changes.