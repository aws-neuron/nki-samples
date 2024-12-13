# Neuron Kernel Interface (NKI) Samples

[AWS Neuron](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/) is the software development kit (SDK) designed for ML chips AWS Trainium and Inferentia: 
purpose built for AI workloads. 
At the core of the Neuron SDK is the Neuron Compiler, which takes computation graphs from frameworks like PyTorch and JAX and converts 
them into highly optimized machine code. 

[NKI](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki) is a Python-based programming environment designed for the compiler which
adopts commonly used NumPy andTriton-like syntax along with tile-level semantics. 
NKI also interoperates with the Neuron Profiler, providing insights into performance bottlenecks and instruction latencies. 
It offers tensor printing support, standard error messaging, and built-in kernel simulation capabilities for efficient debugging purposes. 
NKI offers two types of programming interfaces: 
NKI Language ([nki.language](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/api/nki.language.html)) and 
NKI Instruction Set Architecture ([nki.isa](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/api/nki.isa.html)), 
enabling bare-metal access to the chip for full control.

![alt "High-level flow of NKI in the Neuron Compiler. NKI emits IR immediately before the backend-IR compilation stage"](doc_assets/high-level-nki-flow.png#center "High-Level NKI Flow")

### nki.language 
**nki.language** enables precise control over computation and data movement on NeuronCores-- the processing units within AWS Inferentia and Trainium chips. 
Developers can control data movement between device memory and on-chip memory explicitly using `nl.load()` and `nl.store()` operations. 
Developers can then perform the desired computations on loaded tensors, such as element-wise operations or tensor contractions, 
providing crucial performance improvements. Additionally, developers can control how computation is performed on different compute engines inside NeuronCores. 
nki.language APIs are considered high-level APIs and are designed for "ease of use" for ML practitioners. 
To achieve the best performance, developers can enlist the nki.isa APIs.

![alt "Diagram of the NeuronCore Architecture. It shows 4 engines: tensor, vector, scalar, and GPSIMD, connected to SBUF memory. The tensor, vector, and scalar engines are also connected to a high-speed PSUM memory bank that supports accumulate on write. Lastly the HBM (DRAM) is connected to both SBUF and PSUM memory banks."](doc_assets/pm-nc.png#scale_50#center "NeuronCore Architecture")

### nki.isa

**nki.isa** provides direct access to chip instructions to offer flexibility and fine-grained control over instruction usage and performance optimizations. 
Developers can utilize various `nki.isa` instructions using the Tensor, Vector, Scalar, GP-SIMD, and DMA engines. 
For example, developers can use `nki.isa.nc_matmul()` to compute a matrix multiplication using Tensor Engine. 
Alternatively, developers can use `nki.isa.activation()` to apply an activation function on every element of the input tile using Scalar Engine.

## Repository Structure

### src

#### reference
The [reference kernels](src/reference/) are optimized reference kernels. All kernels located in this folder must have all of numeric accuracy tests 
and performance benchmarks defined in the [test](test/) directory. We also demonstrate using these kernels end-to-end in our [integration tests](test/integration/).


#### tutorials
The [tutorial kernels](src/tutorials/) are for educational purpose and include the kernels that are used in NKI guides. 
You can clone these sample kernels and run them directly while reading through the 
[NKI documentation](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/tutorials.html). These kernels are not necessarily high-performance, 
but contain detailed inline comments and have accompanying documentation. 

### test

#### unit
The [unit tests](test/unit) directory contains unit tests and micro-benchmarks for standalone kernels. They run across multiple possible configurations, 
verify the numeric accuracy of the operation, and publish performance results to the [micro-benchmark](docs/benchmarks/micro-benchmark/) results.

#### integration
The [integration tests](tests/integration) folder contains integration tests of (selected) kernels. They verify the numeric accuracy of the model’s output, 
and publish end-to-end performance results into the [integration benchmarks](docs/benchmarks/integration) folder.

## Documentation
The latest NKI documentation can be found on the AWS Documentation site, [here](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/). 
Documentation for NKI kernels are both inline (docstring) and available on the documentation site's 
[kernel API reference page](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/api/nki.kernels.html).

## Versioning
NKI is currently released as **beta** while we gather feedback from our users and integrate it into the API. We will also be updating the NKI API as needed 
to support new Neuron and Neuron Compiler features. While NKI is in beta we may need to make backwards-incompatible changes to incorporate feedback from 
our users or to support new use-cases of NKI on Neuron devices. Upon releasing NKI as generally available (GA), we will commit to not making backwards 
incompatible changes to the NKI API for any supported version of the Neuron compiler. 

## Contributing
We invite you to join the NKI community! If you'd like to share kernels you create with the community, we welcome your contributions to this repository via. 
GitHub pull-requests as well as through filed issues discussing features, bug fixes, new use-cases, and API improvements.

### Getting Help
Have a look at the GitHub issues for this repository where you will find past issues customers have encountered with workarounds and clarifications. 
If you cannot find a suitable issue for your use-case feel free to file an issue asking for assistance or to suggest improvements.

In addition, extensive NKI documentation can be found [here](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki).

### Testing and Merging
Running the binaries for a NKI kernel require Neuron devices on an AWS EC2 instance from trn1, trn1n, or inf2 instance families. 
Details on setting up an instance can be found in [here](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-setup.html).

Before merging, the Neuron team will need to internally test and verify kernels work as expected. If the change is accepted, 
we will manually merge your changes, and it will be merged here upon the next release. 

If you would like to test your kernel without a requiring a Neuron device, you can use `nki.simulate()` to run your kernel using `NumPy` tensors and types. 
An example can be found in the [layernorm tutorial test](test/unit/test_tutorials_layernorm.py).

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

## Licensing
This repository is licensed under the terms of the [MIT-0 License](LICENSE.txt)