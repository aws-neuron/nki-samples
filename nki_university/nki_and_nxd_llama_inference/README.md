# TinyLLama inference with NeuronX Distributed and Neuron Kernel Interface
In this example you can test [TinyLlama](https://huggingface.co/TinyLlama) from Hugging Face on AWS Trainium. This example was built on a trn1.2xlarge instance using this AMI: Deep Learning AMI Neuron (Ubuntu 22.04) 20240927.

This example pulls largely from the Llama2 inference example from NeuronX Distributed available [here](https://github.com/aws-neuron/neuronx-distributed/tree/main/examples/inference/llama2). However, it adds support for 1/ TinyLlama and 2/ Neuron Kernel Interface (NKI).

### Setup
To run this example, first clone the repository with `git clone https://github.com/aws-neuron/nki-samples.git`.

Next, `cd` into `nki_samples/nki_university/nki_and_nxd_llama_inference`.

Then install the requirements with `pip install -r requirements.txt`. 

### Download the model

You'll need to download the TinyLlama model from Hugging Face. You can do this through the `transformers` SDK like this. 

```
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama_v1.1")
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama_v1.1")
```

After that, save the model into a local directory. This needs to be the same directory you set at the top of `run_llama.py`.

```
model.save_pretrained('/home/ubuntu/models/Tiny-Llama')
tokenizer.save_pretrained('/home/ubuntu/models/Tiny-Llama')
```

### Test the script
Once you've installed all the packages and downloaded your model, you should be ready to test the script. This is done with `python run_llama.py`. 

This script will take at least 30 minutes to complete because it does the following: 1/ compiles your model 2/ loads to Neuron device 3/ tests on Neuron 4/ compares accuracy 5/ runs benchmark suite.

### Write your NKI kernel
Your NKI kernels can operate like normal Python functions inside of this project, such as within `llama2/neuron_modeling_llama.py`. Your script already has a sample kernel, `nki_tensor_add_`, which simply takes the addition of the hidden and residual states during the forward pass. This is available in `llama2/neuron_modeling_llama.py`. This kernel has been tested and confirmed for both accuracy and performance. 


