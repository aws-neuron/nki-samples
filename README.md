# Neuron Kernel Interface (NKI) Samples

[AWS Neuron](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/) is the software development kit (SDK) designed for ML chips AWS Trainium and Inferentia:
purpose built for AI workloads.
At the core of the Neuron SDK is the Neuron Compiler, which takes computation graphs from frameworks like PyTorch and JAX and converts
them into highly optimized machine code.

[NKI](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki) is a Python-based programming environment designed for the compiler which
adopts commonly used NumPy and Triton-like syntax along with tile-level semantics.
NKI also interoperates with the Neuron Profiler, providing insights into performance bottlenecks and instruction latencies.
It offers tensor printing support, standard error messaging, and built-in kernel simulation capabilities for efficient debugging purposes.
NKI offers two types of programming interfaces:
NKI Language ([nki.language](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/api/nki.language.html)) and
NKI Instruction Set Architecture ([nki.isa](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/api/nki.isa.html)),
enabling bare-metal access to the chip for full control.

![alt "High-level flow of NKI in the Neuron Compiler. NKI emits IR immediately before the backend-IR compilation stage"](doc_assets/high-level-nki-flow.png#center "High-Level NKI Flow")

## Documentation
The latest NKI documentation can be found on the AWS Documentation site, [here](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/).
Documentation for NKI kernels are both inline (docstring) and available on the documentation site's
[kernel API reference page](https://aws-neuron.github.io/nki-samples/).

## Repository Structure

### src

#### reference
This folder contains the source code of the `neuronxcc.nki.kernels`, and they are optimized kernels from the Neuron Team serving as samples.

All kernels located in this folder have numeric accuracy tests
and performance benchmarks defined in the [test](test/) directory. We also demonstrate using these kernels end-to-end in our [integration tests](test/integration/).

Note that these kernels are already being deployed as part of the Neuron stack. With flash attention as an example,
[compiling Llama models with transformers-neuronx](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/transformers-neuronx/transformers-neuronx-developer-guide.html)
will automatically invoke the `flash_fwd` kernel in [attention.py](src/nki_samples/reference/attention.py). Therefore, replacing the framework operators with these NKI kernels likely won't result in extra performance benefit.


#### tutorials
The [tutorial kernels](src/nki_samples/tutorials/) are for educational purpose and include the kernels that are used in NKI guides.
You can clone these sample kernels and run them directly while reading through the
[NKI documentation](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/tutorials.html). These kernels are not necessarily high-performance,
but contain detailed inline comments and have accompanying documentation.

### contributed

The [contributed](contributed/) directory contains experimental and advanced NKI kernels from both community contributors and the AWS Neuron team. These kernels showcase cutting-edge optimizations and specialized implementations that demonstrate advanced NKI programming techniques beyond what's available in the reference kernels.

**Important Notice**: All kernels in this directory:
- Have been ested only against internal nightly builds
- May not be compatible with public NeuronSDK releases
- Have not been extensively tested across all input configurations
- Carry no compatibility guarantees
- Behavior may be modified without prior notice

### test

#### unit
The [unit tests](test/unit) directory contains unit tests and micro-benchmarks for standalone kernels. They run across multiple possible configurations,
verify the numeric accuracy of the operation, and publish performance results to the [micro-benchmark](docs/benchmarks/micro-benchmark/) results.

#### integration
The [integration tests](tests/integration) folder contains integration tests of (selected) kernels. They verify the numeric accuracy of the model’s output,
and publish end-to-end performance results into the [integration benchmarks](docs/benchmarks/integration) folder.

## Maintenance Policy
NKI is currently released as **beta** while we gather feedback from our users and integrate it into the API. NKI API follow the [Neuron SDK Maintenance Policy](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/sdk-policy.html).

## Getting Help
Have a look at the GitHub issues for this repository where you will find past issues customers have encountered with workarounds and clarifications.
If you cannot find a suitable issue for your use-case feel free to [file an issue](https://github.com/aws-neuron/nki-samples/issues/new) to ask for assistance or to suggest improvements. Please read [CONTRIBUTING.md](CONTRIBUTING.md) for detailed information on submitting issues.

## Contributing
We invite you to join the NKI community! If you'd like to share kernels you create with the community, we welcome your contributions to this repository via
GitHub pull-requests as well as through filed issues discussing features, bug fixes, new use-cases, and API improvements. Please see [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Licensing
This repository is licensed under the terms of the [MIT-0 License](LICENSE.txt)
