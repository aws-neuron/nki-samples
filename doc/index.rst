NKI Samples
==============

.. currentmodule:: nki_samples.reference

.. _nki_kernels:

nki_samples.reference
---------------------

All kernels located in this folder have numeric accuracy tests and 
performance benchmarks defined in the test directory. We also demonstrate 
using these kernels end-to-end in our integration tests.

You are welcome to customize them to fit your unique workloads, and contributing to the repository by opening a PR. 
Note that these kernels are already being deployed as part of the Neuron stack. With flash attention as an example,
`compiling Llama models with transformers-neuronx <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/transformers-neuronx/transformers-neuronx-developer-guide.html>`_
will automatically invoke the `flash_fwd` kernel listed here. Therefore, replacing the framework operators with these 
NKI kernels likely won't result in extra performance benefit.

Please see the `README <https://github.com/aws-neuron/nki-samples>`_ page 
of the GitHub Repository `nki-samples <https://github.com/aws-neuron/nki-samples>`_ for more details.

For NKI documentation, please refer to the main `Neuron SDK documentation page <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/index.html>`_.

Relationship to `neuronxcc.nki.kernels`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The kernels under `reference` folder is also available in the `neuronxcc.nki.kernels` namespace. The 
kernels in the `neuronxcc` is synced with this repository on every Neuron SDK release. 


.. toctree::
    :maxdepth: 2
    
    nki_samples.reference.attention
    nki_samples.reference.vision
    nki_samples.reference.allocated_fused_linear
    nki_samples.reference.allocated_attention


nki_samples.tutorial
---------------------

Please refer to `this page <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/tutorials.html>`_ for the 
tutorials. The code associated with the tutorial can be found at `nki-samples/src/tutorials <https://github.com/aws-neuron/nki-samples/tree/main/src/tutorials>`_