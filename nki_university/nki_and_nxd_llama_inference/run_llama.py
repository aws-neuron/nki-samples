import torch
from llama2.llama2_runner import LlamaRunner
from transformers import GenerationConfig
import os

model_path = "/home/ubuntu/models/Tiny-Llama/"
traced_model_path = "/home/ubuntu/models/Tiny-Llama-traced/"

torch.manual_seed(0)

def llama_sample():
    # Compile the model for a specific configuration
    generation_config = GenerationConfig.from_pretrained(model_path)
    generation_config.top_k = 1
    generation_config.do_sample = True

    runner = LlamaRunner(model_path=model_path, tokenizer_path=model_path)
    
    batch_size = 2
    max_prompt_length = 1024
    sequence_length = 2048

    runner.trace(
        traced_model_path=traced_model_path,
        tp_degree=2,
        batch_size=batch_size,
        max_prompt_length=max_prompt_length,
        sequence_length=sequence_length,
        on_device_sampling=True,
    )

    # Load model weights into Neuron device
    # We will use the returned model to run accuracy and perf tests
    print("\nLoading model to Neuron device ..")
    neuron_model = runner.load_neuron_model(traced_model_path)

    # Confirm the traced model matches the huggingface model run on cpu
    print("\nChecking accuracy ..")
    runner.check_accuracy(neuron_model, batch_size, sequence_length)

    # Perform inference
    prompt = ["I believe the meaning of life is", "The color of the sky is"]
    print("\nGenerating ..")
    _, outputs = runner.generate_on_neuron(prompt, neuron_model)
    print("Generated outputs:")
    for idx, output in enumerate(outputs):
        print(f"output {idx}: {output}")

    print("\nBenchmarking ..")
    # Now lets benchmark
    runner.benchmark_sampling(neuron_model)


if __name__ == "__main__":
    llama_sample()
