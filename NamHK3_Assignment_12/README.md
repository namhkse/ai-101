# Satellite Image Classification with Azure OpenAI 🌤️

This project demonstrates how to use **Azure OpenAI** with **LangChain** to classify satellite images as either **Clear** or **Cloudy**.  

## 🔹 What the code does
1. **Setup** – Loads environment variables for Azure OpenAI configuration.  
2. **LLM Initialization** – Connects to an Azure OpenAI ChatGPT model with structured output using a `Pydantic` schema (`WeatherResponse`).  
3. **Image Processing** – Downloads a sample satellite image from a URL and encodes it in **base64**.  
4. **Prompt Construction** – Sends the image and instructions to the model to classify the weather condition.  
5. **Prediction** – The model returns:  
   - `result`: `"Clear"` or `"Cloudy"`  
   - `accuracy`: confidence score of the classification.  

## 🔹 Example Output

### 🖼️ Example 1: Cloudy Sky
#### Input image:
![alt text](https://images.pexels.com/photos/53594/blue-clouds-day-fluffy-53594.jpeg)
#### Output
```
Prediction: Cloudy
Accuracy:: 90.0 %
```

### 🖼️ Example 2: Clear Sky
#### Input image:
![alt text](https://images.pexels.com/photos/1775862/pexels-photo-1775862.jpeg)
#### Output
```
Prediction: Clear
Accuracy:: 95.0 %
```

### 🖼️ Example 3: Partially Cloudy
#### Input image:
![alt text](https://images.pexels.com/photos/1459495/pexels-photo-1459495.jpeg)
#### Output
```
Prediction: Clear
Accuracy:: 95.0 %
```

⚡ In short: this code shows how to combine **Azure OpenAI + LangChain structured output + image input** to build an AI-powered **weather scene classifier**.
