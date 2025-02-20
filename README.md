# Phi-3-medium-128k-instruct
## Model Summary

The Phi-3-Medium-128K-Instruct is a 14B parameters, lightweight, state-of-the-art open model trained with the Phi-3 datasets that includes both synthetic data and the filtered publicly available websites data with a focus on high-quality and reasoning dense properties. The model belongs to the Phi-3 family with the Medium version in two variants [4k](https://huggingface.co/microsoft/Phi-3-medium-4k-instruct) and [128K](https://huggingface.co/microsoft/Phi-3-medium-128k-instruct) which is the context length (in tokens) that it can support.

The model has underwent a post-training process that incorporates both supervised fine-tuning and direct preference optimization for the instruction following and safety measures. When assessed against benchmarks testing common sense, language understanding, math, code, long context and logical reasoning, Phi-3-Medium-128K-Instruct showcased a robust and state-of-the-art performance among models of the same-size and next-size-up.

Resources and Technical Documentation:

- [Phi-3 Microsoft Blog](https://aka.ms/Phi-3Build2024)
- [Phi-3 Technical Report](https://aka.ms/phi3-tech-report)
- [Phi-3 on Azure AI Studio](https://aka.ms/phi3-azure-ai)
- [Phi-3 Cookbook](https://github.com/microsoft/Phi-3CookBook)

|        | Short Context                                                | Long Context                                                 |
| ------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Mini   | 4K [[HF\]](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) ; [[ONNX\]](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx) ; [[GGUF\]](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf) | 128K [[HF\]](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct) ; [[ONNX\]](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct-onnx) |
| Small  | 8K [[HF\]](https://huggingface.co/microsoft/Phi-3-small-8k-instruct) ; [[ONNX\]](https://huggingface.co/microsoft/Phi-3-small-8k-instruct-onnx-cuda) | 128K [[HF\]](https://huggingface.co/microsoft/Phi-3-small-128k-instruct) ; [[ONNX\]](https://huggingface.co/microsoft/Phi-3-small-128k-instruct-onnx-cuda) |
| Medium | 4K [[HF\]](https://huggingface.co/microsoft/Phi-3-medium-4k-instruct) ; [[ONNX\]](https://huggingface.co/microsoft/Phi-3-medium-4k-instruct-onnx-cuda) | 128K [[HF\]](https://huggingface.co/microsoft/Phi-3-medium-128k-instruct) ; [[ONNX\]](https://huggingface.co/microsoft/Phi-3-medium-128k-instruct-onnx-cuda) |
| Vision |                                                              | 128K [[HF\]](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct) ; [[ONNX\]](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct-onnx-cuda) |

## Intended Uses

**Primary use cases**

The model is intended for broad commercial and research use in English. The model provides uses for general purpose AI systems and applications which require :

1. Memory/compute constrained environments
2. Latency bound scenarios
3. Strong reasoning (especially code, math and logic)

Our model is designed to accelerate research on language and multimodal models, for use as a building block for generative AI powered features.

**Use case considerations**

Our models are not specifically designed or evaluated for all downstream purposes. Developers should consider common limitations of language models as they select use cases, and evaluate and mitigate for accuracy, safety, and fariness before using within a specific downstream use case, particularly for high risk scenarios. Developers should be aware of and adhere to applicable laws or regulations (including privacy, trade compliance laws, etc.) that are relevant to their use case.

Nothing contained in this Model Card should be interpreted as or deemed a restriction or modification to the license the model is released under.

## How to Use

Phi-3-Medium-128k-Instruct has been integrated in the development version (4.40.2) of `transformers`. Until the official version is released through `pip`, ensure that you are doing one of the following:

- When loading the model, ensure that `trust_remote_code=True` is passed as an argument of the `from_pretrained()` function.
- Update your local `transformers` to the development version: `pip uninstall -y transformers && pip install git+https://github.com/huggingface/transformers`. The previous command is an alternative to cloning and installing from the source.

The current `transformers` version can be verified with: `pip list | grep transformers`.

Phi-3-Medium-128k-Instruct is also available in [Azure AI Studio](https://aka.ms/phi3-azure-ai).

### Tokenizer

Phi-3-Medium-128k-Instruct supports a vocabulary size of up to `32064` tokens. The [tokenizer files](https://huggingface.co/microsoft/Phi-3-medium-128k-instruct/blob/main/added_tokens.json) already provide placeholder tokens that can be used for downstream fine-tuning, but they can also be extended up to the model's vocabulary size.

### Chat Format

Given the nature of the training data, the Phi-3-Medium-128k-Instruct model is best suited for prompts using the chat format as follows. You can provide the prompt as a question with a generic template as follow:

```markdown
<|user|>\nQuestion <|end|>\n<|assistant|>
```

For example:

```markdown
<|user|>
How to explain Internet for a medieval knight?<|end|>
<|assistant|>
```

where the model generates the text after `<|assistant|>` . In case of few-shots prompt, the prompt can be formatted as the following:

```markdown
<|user|>
I am going to Paris, what should I see?<|end|>
<|assistant|>
Paris, the capital of France, is known for its stunning architecture, art museums, historical landmarks, and romantic atmosphere. Here are some of the top attractions to see in Paris:\n\n1. The Eiffel Tower: The iconic Eiffel Tower is one of the most recognizable landmarks in the world and offers breathtaking views of the city.\n2. The Louvre Museum: The Louvre is one of the world's largest and most famous museums, housing an impressive collection of art and artifacts, including the Mona Lisa.\n3. Notre-Dame Cathedral: This beautiful cathedral is one of the most famous landmarks in Paris and is known for its Gothic architecture and stunning stained glass windows.\n\nThese are just a few of the many attractions that Paris has to offer. With so much to see and do, it's no wonder that Paris is one of the most popular tourist destinations in the world."<|end|>
<|user|>
What is so great about #1?<|end|>
<|assistant|>
```

### Sample inference code

This code snippets show how to get quickly started with running the model on a GPU:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

torch.random.manual_seed(0)
model_id = "/models/Phi-3-medium-128k-instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda", 
    torch_dtype="auto", 
    trust_remote_code=False, 
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [
    {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"},
    {"role": "assistant", "content": "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey."},
    {"role": "user", "content": "What about solving an 2x + 3 = 7 equation?"},
]

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

generation_args = {
    "max_new_tokens": 500,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
}

output = pipe(messages, **generation_args)
print(output[0]['generated_text'])
```

*Some applications/frameworks might not include a BOS token (`<s>`) at the start of the conversation. Please ensure that it is included since it provides more reliable results.*

###  code output
```bash
Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:08<00:00,  1.49s/it]
Device set to use cuda
/usr/local/anaconda3/lib/python3.12/site-packages/transformers/generation/configuration_utils.py:629: UserWarning: `do_sample` is set to `False`. However, `temperature` is set to `0.0` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `temperature`.
  warnings.warn(
 To solve the equation 2x + 3 = 7, you need to isolate the variable x. Here are the steps:

1. Subtract 3 from both sides of the equation: 2x + 3 - 3 = 7 - 3. This simplifies to 2x = 4.
2. Divide both sides of the equation by 2: (2x)/2 = 4/2. This simplifies to x = 2.

So, the solution to the equation 2x + 3 = 7 is x = 2.

```
## Responsible AI Considerations

Like other language models, the Phi series models can potentially behave in ways that are unfair, unreliable, or offensive. Some of the limiting behaviors to be aware of include:

- Quality of Service: the Phi models are trained primarily on English text. Languages other than English will experience worse performance. English language varieties with less representation in the training data might experience worse performance than standard American English.
- Representation of Harms & Perpetuation of Stereotypes: These models can over- or under-represent groups of people, erase representation of some groups, or reinforce demeaning or negative stereotypes. Despite safety post-training, these limitations may still be present due to differing levels of representation of different groups or prevalence of examples of negative stereotypes in training data that reflect real-world patterns and societal biases.
- Inappropriate or Offensive Content: these models may produce other types of inappropriate or offensive content, which may make it inappropriate to deploy for sensitive contexts without additional mitigations that are specific to the use case.
- Information Reliability: Language models can generate nonsensical content or fabricate content that might sound reasonable but is inaccurate or outdated.
- Limited Scope for Code: Majority of Phi-3 training data is based in Python and use common packages such as "typing, math, random, collections, datetime, itertools". If the model generates Python scripts that utilize other packages or scripts in other languages, we strongly recommend users manually verify all API uses.

Developers should apply responsible AI best practices and are responsible for ensuring that a specific use case complies with relevant laws and regulations (e.g. privacy, trade, etc.). Important areas for consideration include:

- Allocation: Models may not be suitable for scenarios that could have consequential impact on legal status or the allocation of resources or life opportunities (ex: housing, employment, credit, etc.) without further assessments and additional debiasing techniques.
- High-Risk Scenarios: Developers should assess suitability of using models in high-risk scenarios where unfair, unreliable or offensive outputs might be extremely costly or lead to harm. This includes providing advice in sensitive or expert domains where accuracy and reliability are critical (ex: legal or health advice). Additional safeguards should be implemented at the application level according to the deployment context.
- Misinformation: Models may produce inaccurate information. Developers should follow transparency best practices and inform end-users they are interacting with an AI system. At the application level, developers can build feedback mechanisms and pipelines to ground responses in use-case specific, contextual information, a technique known as Retrieval Augmented Generation (RAG).
- Generation of Harmful Content: Developers should assess outputs for their context and use available safety classifiers or custom solutions appropriate for their use case.
- Misuse: Other forms of misuse such as fraud, spam, or malware production may be possible, and developers should ensure that their applications do not violate applicable laws and regulations.

## Training

### Model

- Architecture: Phi-3-Medium-128k-Instruct has 14B parameters and is a dense decoder-only Transformer model. The model is fine-tuned with Supervised fine-tuning (SFT) and Direct Preference Optimization (DPO) to ensure alignment with human preferences and safety guidlines.
- Inputs: Text. It is best suited for prompts using chat format.
- Context length: 128k tokens
- GPUs: 512 H100-80G
- Training time: 42 days
- Training data: 4.8T tokens
- Outputs: Generated text in response to the input
- Dates: Our models were trained between February and April 2024
- Status: This is a static model trained on an offline dataset with cutoff date October 2023. Future versions of the tuned models may be released as we improve models.
- Release dates: The model weight is released on May 21, 2024.

### Datasets

Our training data includes a wide variety of sources, totaling 4.8 trillion tokens (including 10% multilingual), and is a combination of

1. Publicly available documents filtered rigorously for quality, selected high-quality educational data, and code;
2. Newly created synthetic, “textbook-like” data for the purpose of teaching math, coding, common sense reasoning, general knowledge of the world (science, daily activities, theory of mind, etc.);
3. High quality chat format supervised data covering various topics to reflect human preferences on different aspects such as instruct-following, truthfulness, honesty and helpfulness.

We are focusing on the quality of data that could potentially improve the reasoning ability for the model, and we filter the publicly available documents to contain the correct level of knowledge. As an example, the result of a game in premier league in a particular day might be good training data for frontier models, but we need to remove such information to leave more model capacity for reasoning for the small size models. More details about data can be found in the [Phi-3 Technical Report](https://aka.ms/phi3-tech-report).

## Benchmarks

We report the results for Phi-3-Medium-128k-Instruct on standard open-source benchmarks measuring the model's reasoning ability (both common sense reasoning and logical reasoning). We compare to Mixtral-8x22b, Gemini-Pro, Command R+ 104B, Llama-3-70B-Instruct, GPT-3.5-Turbo-1106, and GPT-4-Turbo-1106(Chat).

All the reported numbers are produced with the exact same pipeline to ensure that the numbers are comparable. These numbers might differ from other published numbers due to slightly different choices in the evaluation.

As is now standard, we use few-shot prompts to evaluate the models, at temperature 0. The prompts and number of shots are part of a Microsoft internal tool to evaluate language models, and in particular we did no optimization to the pipeline for Phi-3. More specifically, we do not change prompts, pick different few-shot examples, change prompt format, or do any other form of optimization for the model.

The number of k–shot examples is listed per-benchmark.

| Benchmark                     | Phi-3-Medium-128k-Instruct 14b | Command R+ 104B | Mixtral 8x22B | Llama-3-70B-Instruct | GPT3.5-Turbo version 1106 | Gemini Pro | GPT-4-Turbo version 1106 (Chat) |
| ----------------------------- | ------------------------------ | --------------- | ------------- | -------------------- | ------------------------- | ---------- | ------------------------------- |
| AGI Eval 5-shot               | 49.7                           | 50.1            | 54.0          | 56.9                 | 48.4                      | 49.0       | 59.6                            |
| MMLU 5-shot                   | 76.6                           | 73.8            | 76.2          | 80.2                 | 71.4                      | 66.7       | 84.0                            |
| BigBench Hard 3-shot          | 77.9                           | 74.1            | 81.8          | 80.4                 | 68.3                      | 75.6       | 87.7                            |
| ANLI 7-shot                   | 57.3                           | 63.4            | 65.2          | 68.3                 | 58.1                      | 64.2       | 71.7                            |
| HellaSwag 5-shot              | 81.6                           | 78.0            | 79.0          | 82.6                 | 78.8                      | 76.2       | 88.3                            |
| ARC Challenge 10-shot         | 91.0                           | 86.9            | 91.3          | 93.0                 | 87.4                      | 88.3       | 95.6                            |
| ARC Easy 10-shot              | 97.6                           | 95.7            | 96.9          | 98.2                 | 96.3                      | 96.1       | 98.8                            |
| BoolQ 2-shot                  | 86.5                           | 86.1            | 82.7          | 89.1                 | 79.1                      | 86.4       | 91.3                            |
| CommonsenseQA 10-shot         | 82.2                           | 82.0            | 82.0          | 84.4                 | 79.6                      | 81.8       | 86.7                            |
| MedQA 2-shot                  | 67.6                           | 59.2            | 67.9          | 78.5                 | 63.4                      | 58.2       | 83.7                            |
| OpenBookQA 10-shot            | 87.2                           | 86.8            | 88.6          | 91.8                 | 86.0                      | 86.4       | 93.4                            |
| PIQA 5-shot                   | 87.8                           | 86.4            | 85.0          | 85.3                 | 86.6                      | 86.2       | 90.1                            |
| Social IQA 5-shot             | 79.0                           | 75.3            | 78.2          | 81.1                 | 68.3                      | 75.4       | 81.7                            |
| TruthfulQA (MC2) 10-shot      | 74.3                           | 57.8            | 67.4          | 81.9                 | 67.7                      | 72.6       | 85.2                            |
| WinoGrande 5-shot             | 78.9                           | 77.0            | 75.3          | 83.3                 | 68.8                      | 72.2       | 86.7                            |
| TriviaQA 5-shot               | 73.9                           | 82.8            | 84.5          | 78.5                 | 85.8                      | 80.2       | 73.3                            |
| GSM8K Chain of Thought 8-shot | 87.5                           | 78.3            | 83.8          | 93.5                 | 78.1                      | 80.4       | 94.2                            |
| HumanEval 0-shot              | 58.5                           | 61.6            | 39.6          | 78.7                 | 62.2                      | 64.4       | 79.9                            |
| MBPP 3-shot                   | 73.8                           | 68.9            | 70.7          | 81.3                 | 77.8                      | 73.2       | 86.7                            |
| Average                       | 77.3                           | 75.0            | 76.3          | 82.5                 | 74.3                      | 75.4       | 85.2                            |

We take a closer look at different categories across 80 public benchmark datasets at the table below:

| Benchmark                    | Phi-3-Medium-128k-Instruct 14b | Command R+ 104B | Mixtral 8x22B | Llama-3-70B-Instruct | GPT3.5-Turbo version 1106 | Gemini Pro | GPT-4-Turbo version 1106 (Chat) |
| ---------------------------- | ------------------------------ | --------------- | ------------- | -------------------- | ------------------------- | ---------- | ------------------------------- |
| Popular aggregated benchmark | 72.3                           | 69.9            | 73.4          | 76.3                 | 67.0                      | 67.5       | 80.5                            |
| Reasoning                    | 83.2                           | 79.3            | 81.5          | 86.7                 | 78.3                      | 80.4       | 89.3                            |
| Language understanding       | 75.3                           | 75.7            | 78.7          | 77.9                 | 70.4                      | 75.3       | 81.6                            |
| Code generation              | 64.2                           | 68.6            | 60.0          | 69.3                 | 70.4                      | 66.7       | 76.1                            |
| Math                         | 52.9                           | 45.3            | 52.5          | 59.7                 | 52.8                      | 50.9       | 67.1                            |
| Factual knowledge            | 47.5                           | 60.3            | 60.6          | 52.4                 | 63.4                      | 54.6       | 45.9                            |
| Multilingual                 | 62.2                           | 67.8            | 69.8          | 62.0                 | 67.0                      | 73.4       | 78.2                            |
| Robustness                   | 70.2                           | 57.9            | 65.5          | 78.7                 | 69.3                      | 69.7       | 84.6                            |

## Software

- [PyTorch](https://github.com/pytorch/pytorch)
- [DeepSpeed](https://github.com/microsoft/DeepSpeed)
- [Transformers](https://github.com/huggingface/transformers)
- [Flash-Attention](https://github.com/HazyResearch/flash-attention)

## Hardware

Note that by default, the Phi-3-Medium model uses flash attention, which requires certain types of GPU hardware to run. We have tested on the following GPU types:

- NVIDIA A100
- NVIDIA A6000
- NVIDIA H100

If you want to run the model on:

- Optimized inference on GPU, CPU, and Mobile: use the **ONNX** models [128k](https://huggingface.co/microsoft/Phi-3-medium-128k-instruct-onnx-cuda)

## Cross Platform Support

ONNX runtime ecosystem now supports Phi3 Medium models across platforms and hardware. Optimized phi-3 models are also published here in ONNX format, to run with ONNX Runtime on CPU and GPU across devices, including server platforms, Windows, Linux and Mac desktops, and mobile CPUs, with the precision best suited to each of these targets. DirectML GPU acceleration is supported for Windows desktops GPUs (AMD, Intel, and NVIDIA).
Along with DML, ONNX Runtime provides cross platform support for Phi3 Medium across a range of devices CPU, GPU, and mobile. Here are some of the optimized configurations we have added:

1. ONNX models for int4 DML: Quantized to int4 via AWQ
2. ONNX model for fp16 CUDA
3. ONNX model for int4 CUDA: Quantized to int4 via RTN
4. ONNX model for int4 CPU and Mobile: Quantized to int4 via RTN

## License

The model is licensed under the [MIT license](https://huggingface.co/microsoft/Phi-3-medium-128k/resolve/main/LICENSE).

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow [Microsoft’s Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks). Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party’s policies.
