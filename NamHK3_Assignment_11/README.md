# AI Assistant with LangChain, Azure OpenAI, and Tools

This script demonstrates how to build a simple **AI assistant** using **LangChain** and **LangGraph ReAct Agent** with external tools.  

## 🔹 What it does
1. **Weather Tool** – Uses the `OpenWeatherMapAPIWrapper` to fetch real-time weather for any city.  
2. **Web Search Tool** – Uses `TavilySearch` to retrieve the latest online information.  
3. **LLM Integration** – Connects to **Azure OpenAI ChatGPT model** as the reasoning engine.  
4. **Agent Setup** – Combines the tools and LLM into a **ReAct Agent** that can decide when to call each tool.  
5. **Conversation Loop** – Processes a list of mock user questions (weather, AI news, sports) and returns answers.  

## 🔹 Example Output
```sh
Welcome to the AI assistant. Type 'exit' to stop.
User:  What's the weather in Hanoi?
 get_weather tool caliing: Getting weather for Hanoi
AI:  The current weather in Hanoi is as follows:

- **Temperature:** 29.0°C (84.2°F)
- **Condition:** Partly cloudy
- **Wind Speed:** 2.2 mph (3.6 kph) from the NNE
- **Humidity:** 75%
- **Feels Like:** 34.4°C (93.9°F)
- **Pressure:** 1006 mb
- **Visibility:** 10 km

For more details, you can check [Weather API](https://www.weatherapi.com/).
User:  Tell me about the latest news in AI.
AI:  Here are some of the latest news and updates in AI:

1. **Former Twitter CEO Parag Agrawal's AI Startup:** Agrawal's startup, Parallel, has reportedly beaten GPT-5 in deep web research.

2. **Google's New Introduction:** Google has introduced Gemma 3 270M, which is designed for task-specific applications.

For more details, you can read the full article on [Analytics India Magazine](https://analyticsindiamag.com/ai-news-updates/).  
User:  Who won the last World Cup?
AI:  The last FIFA World Cup was held in 2022, and Argentina won the tournament, defeating France in the final. You can find more details about World Cup winners on [Transfermarkt](https://www.transfermarkt.us/world-cup/erfolge/pokalwettbewerb/FIWC).
User:  exit
Goodbye!
```

⚡ In short: this code shows how to integrate **Azure OpenAI + LangChain agents + external APIs** to build a multi-tool conversational assistant.  