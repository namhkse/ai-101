# Build Resume Generation Using LLaMA3 Locally

## Overview
This project demonstrates how to use llama-cpp-python
with a Hugging Face GGUF model (e.g., LLaMA 2), organized with an object-oriented approach in Python.

The code also includes exception handling to make the text generation process more robust.

## 📦 Requirements

- Python 3.9+ (recommended 3.11+)
- Install dependencies:
```
pip install llama-cpp-python huggingface_hub
```

## 📂 Model Setup

This example uses a quantized LLaMA 2 model from Hugging Face:

TheBloke/Llama-2-7B-Chat-GGUF

## Usage


### Sample 1
```py
prompt_text = "Once upon a time,"
print_generated_text(prompt_text)
```

```sh
Generated Text:
 a young woman named Maria lived in a small village nestled in the rolling hills of Tuscany. Unterscheidung between the two regions is not always clear-cut, as both share some common characteristics and are often blurred in their cultural identity.
```

### Sample 2
```py
prompt_text = "Explain the difference between supervised and unsupervised learning."
print_generated_text(prompt_text)
```

```sh
Generated Text:
Unsupervised learning is a type of machine learning where the algorithm tries to find patterns or relationships in the data without any labeled examples. In other words, the algorithm has no information about the correct output or target variable. Unsuper
```

### Sample 3
```py
prompt_text = "Newton’s three laws of motion"
print_generated_text(prompt_text)
```

```sh
Generated Text:
 can be used to describe and predict the motion of objects under various conditions. Unterscheidung zwischen “Law” und “Laws” 1.0 out of 5.0 (1) 10. The Newton's laws of motion
```