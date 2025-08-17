# pip install langchain-openai langchain-community langchain-tavily langgraph pyowm
import os
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_openai import AzureChatOpenAI
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
# Step 1: Setup API keys
load_dotenv()

# Step 2: Define weather tool using Langchain wrapper
weather = OpenWeatherMapAPIWrapper()


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city.
    Args: city (str): The name of the city to get the weather for.
    Returns: str: A string describing the current weather in the specified
    city.
    """
    print(f" get_weather tool caliing: Getting weather for {city}")
    return weather.run(city)


# Step 3: Initialize Tavily search tool
tavily_search_tool = TavilySearch(
    max_results=1,
    topic="general",
)

# Step 4: Initialize Azure OpenAI LLM
llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_DEPLOYMENT_NAME"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),  # or your deployment
    api_version="2024-07-01-preview",  # or your api version
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)

# Step 5: Setup Langchain agent with both tools
tools = [get_weather, tavily_search_tool]
agent = create_react_agent(
    model=llm,
    tools=tools,
)

print("Welcome to the AI assistant. Type 'exit' to stop.")
messages = []

# Mock user questions for automatic input
mock_questions = [
    "What's the weather in Hanoi?",
    "Tell me about the latest news in AI.",
    "Who won the last World Cup?",
    "exit",
]
for user_input in mock_questions:
    print("User: ", user_input)
    if user_input.lower() == "exit":
        print("Goodbye!")
        break
    messages.append({"role": "user", "content": user_input})
    response = agent.invoke({"messages": messages})
    messages.append(
        {"role": "assistant", "content": response["messages"][-1].content})
    print("AI: ", response["messages"][-1].content)
