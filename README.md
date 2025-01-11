---
title: GWQ 2B
emoji: 🔥
colorFrom: pink
colorTo: indigo
sdk: gradio
sdk_version: 5.11.0
app_file: app.py
pinned: true
license: creativeml-openrail-m
short_description: GWQ + Phi-4 o1
---


![gwq2.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/Ayc6YKE6FKYKb8Mible4z.png)


# **GWQ2b - Gemma with Questions2b**

GWQ2b is a family of lightweight, state-of-the-art open models from Google, built using the same research and technology employed to create the Gemini models. These models are text-to-text, decoder-only large language models, available in English, with open weights for both pre-trained and instruction-tuned variants. GWQ2b models are well-suited for a variety of text generation tasks, including question answering, summarization, and reasoning. GWQ2b is fine-tuned on the Chain of Continuous Thought Synthetic Dataset, built upon the Gemma2forCasualLM architecture.

# **Running GWQ2b Demo**

```python
# pip install accelerate
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("prithivMLmods/GWQ2b")
model = AutoModelForCausalLM.from_pretrained(
    "prithivMLmods/GWQ2b",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

input_text = "Write me a poem about Machine Learning."
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids, max_new_tokens=32)
print(tokenizer.decode(outputs[0]))
```

You can ensure the correct chat template is applied by using `tokenizer.apply_chat_template` as follows:
```python
messages = [
    {"role": "user", "content": "Write me a poem about Machine Learning."},
]
input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True).to("cuda")

outputs = model.generate(**input_ids, max_new_tokens=256)
print(tokenizer.decode(outputs[0]))
```
# **Key Architecture**

1. **Transformer-Based Design**:  
   GWQ2b leverages the transformer architecture, utilizing self-attention mechanisms to process input text and capture contextual relationships effectively.

2. **Lightweight and Efficient**:  
   It is designed to be computationally efficient, with fewer parameters compared to larger models, making it ideal for deployment on resource-constrained devices or environments.

3. **Modular Layers**:  
   The architecture consists of modular encoder and decoder layers, allowing flexibility in adapting the model for specific tasks like text generation, summarization, or classification.

4. **Attention Mechanisms**:  
   GWQ2b employs multi-head self-attention to focus on relevant parts of the input text, improving its ability to handle long-range dependencies and complex language structures.

5. **Pre-training and Fine-Tuning**:  
   The model is pre-trained on large text corpora and can be fine-tuned for specific tasks, such as markdown processing in ReadM.Md, to enhance its performance on domain-specific data.

6. **Scalability**:  
   The architecture supports scaling up or down based on the application's requirements, balancing performance and resource usage.

7. **Open-Source and Customizable**:  
   Being open-source, GWQ2b allows developers to modify and extend its architecture to suit specific use cases, such as integrating it into tools like ReadM.Md for markdown-related tasks.

# **Intended Use of GWQ2b (Gemma with Questions2b)**

1. **Question Answering:**  
   The model excels in generating concise and relevant answers to user-provided queries across various domains.

2. **Summarization:**  
   It can be used to summarize large bodies of text, making it suitable for news aggregation, academic research, and report generation.

3. **Reasoning Tasks:**  
   GWQ2b is fine-tuned on the Chain of Continuous Thought Synthetic Dataset, which enhances its ability to perform reasoning, multi-step problem solving, and logical inferences.

4. **Text Generation:**  
   The model is ideal for creative writing tasks such as generating poems, stories, and essays. It can also be used for generating code comments, documentation, and markdown files.

5. **Instruction Following:**  
   GWQ2b’s instruction-tuned variant is suitable for generating responses based on user instructions, making it useful for virtual assistants, tutoring systems, and automated customer support.

6. **Domain-Specific Applications:**  
   Thanks to its modular design and open-source nature, the model can be fine-tuned for specific tasks like legal document summarization, medical record analysis, or financial report generation.

# **Limitations of GWQ2b**

1. **Resource Requirements:**  
   Although lightweight compared to larger models, the 9B parameter size still requires significant computational resources, including GPUs with large memory for inference.

2. **Knowledge Cutoff:**  
   The model’s pre-training data may not include recent information, making it less effective for answering queries on current events or newly developed topics.

3. **Bias in Outputs:**  
   Since the model is trained on publicly available datasets, it may inherit biases present in those datasets, leading to potentially biased or harmful outputs in sensitive contexts.

4. **Hallucinations:**  
   Like other large language models, GWQ2b can occasionally generate incorrect or nonsensical information, especially when asked for facts or reasoning outside its training scope.

5. **Lack of Common-Sense Reasoning:**  
   While GWQ2b is fine-tuned for reasoning, it may still struggle with tasks requiring deep common-sense knowledge or nuanced understanding of human behavior and emotions.

6. **Dependency on Fine-Tuning:**  
   For optimal performance on domain-specific tasks, fine-tuning on relevant datasets is required, which demands additional computational resources and expertise.
  
7. **Context Length Limitation:**  
   The model’s ability to process long documents is limited by its maximum context window size. If the input exceeds this limit, truncation may lead to loss of important information.
   
   
Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
