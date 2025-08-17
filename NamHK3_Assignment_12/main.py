# pip install langchain-openai pillow requests
import base64
import io
import os
from dotenv import load_dotenv
import requests
from PIL import Image
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field

# --- Azure OpenAI Config ---
load_dotenv()

# --- Setup LLM ---
llm = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_DEPLOYMENT_NAME"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version="2024-07-01-preview",
)
# --- Output Schema ---


class WeatherResponse(BaseModel):
    accuracy: float = Field(description="The accuracy of the result")
    result: str = Field(description="The result of the classification")


llm_with_structured_output = llm.with_structured_output(WeatherResponse)
# --- Load image from URL ---
# image_url = "https://images.pexels.com/photos/53594/blue-clouds-day-fluffy-53594.jpeg"  # cloud
# image_url = "https://images.pexels.com/photos/1775862/pexels-photo-1775862.jpeg"        # sun 
image_url = "https://images.pexels.com/photos/1459495/pexels-photo-1459495.jpeg"        # tree

response = requests.get(image_url)
image_bytes = response.content
image_data_base64 = base64.b64encode(image_bytes).decode("utf-8")
# --- Prompt Construction ---
message = [
    {
        "role": "system",
        "content": """Based on the satellite image provided, classify the
scene as either:
 'Clear' (no clouds) or 'Cloudy' (with clouds).
 Respond with only one word: either 'Clear' or 'Cloudy' and Accuracy.
Do not provide explanations.""",
    },
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Classify the scene as either: 'Clear' or 'Cloudy' and Accuracy.",
            },
            {
                "type": "image_url",
                "image_url": {"url":
                              f"data:image/jpeg;base64,{image_data_base64}"},
            },
        ],
    },
]
# --- Call Azure OpenAI ---
try:
    result = llm_with_structured_output.invoke(message)
    print(f"Prediction: {result.result}")
    print(f"Accuracy:: {result.accuracy} %")
except Exception as e:
    print(f"Error: {e}")
