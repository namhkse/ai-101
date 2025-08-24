# prompt: generate python code to use LLaMA cpp and a huggingface model, code must handle the exception and organize as OOP
# !pip install llama-cpp-python
from llama_cpp import Llama
import os

class LlamaModel:
    def __init__(self, n_ctx=512):
        """
        Initializes the Llama model.
        """
        self.llm = Llama.from_pretrained(
            repo_id="TheBloke/Llama-2-7B-Chat-GGUF",
            filename="llama-2-7b-chat.Q2_K.gguf"
        )

    def generate_text(self, prompt, max_tokens=100, temperature=0.7):
        """
        Generates text using the loaded Llama model.
        Args:
        prompt (str): The input prompt for text generation.
        max_tokens (int): The maximum number of tokens to generate.
        temperature (float): The sampling temperature.
        Returns:
        str: The generated text.
        """
        try:
            output = self.llm(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return output['choices'][0]['text']
        except Exception as e:
            raise RuntimeError(f"Failed to generate text: {e}")

def print_generated_text(prompt_text: str):
    try:
        # Initialize the model
        llama_model = LlamaModel()
        # Generate text
        generated_text = llama_model.generate_text(prompt_text, max_tokens=50)
        print("\nGenerated Text:")
        print(generated_text)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the model file exists at the specified path.")
    except RuntimeError as e:
        print(f"Error during model operation: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
