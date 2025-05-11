'''
Author: JimZhang
Date: 2025-05-11 16:17:53
LastEditors: 很拉风的James
LastEditTime: 2025-05-11 16:19:41
FilePath: /Code/Project/JimiAgent/example.py
Description: 

'''
from component.agent_init import Agent
import json

def example_usage():
    """
    Example demonstrating how to use the enhanced Agent.
    """
    import os
    
    # Initialize the agent with your API key
    api_key = os.environ['API_KEY']

    agent = Agent(
        api_key=api_key,
        base_url=os.environ['API_URL'],
        model_name="glm-z1-flash",
        system_prompt="You are a helpful AI assistant with code execution, task planning, and self-debugging capabilities.",
        description="AI Assistant with Enhanced Capabilities"
    )
    
    # Register a custom function
    @agent.register_function(name="get_flight_number", description="查询航班号")
    def get_flight_number(date: str, departure: str, destination: str):
        """
        根据始发地、目的地和日期，查询对应日期的航班号
        Args:
            date: 日期，格式 YYYY-MM-DD
            departure: 始发地
            destination: 目的地
        Returns:
            航班号
        """
        flight_number = {
            "北京": {"上海": "1234", "广州": "8321"},
            "上海": {"北京": "1233", "广州": "8123"},
        }
        return {"flight_number": flight_number[departure][destination]}
    
    @agent.register_function(name="get_ticket_price", description="查询票价")
    def get_ticket_price(date: str, flight_number: str):
        """
        查询某航班在某日的票价
        Args:
            date: 日期，格式 YYYY-MM-DD
            flight_number: 航班号
        Returns:
            票价
        """
        return {"ticket_price": "1000"}
    
    @agent.register_function(name="get_weather", description="Get weather for a location")
    def get_weather(location: str, unit: str = "celsius"):
        """
        Get weather information for a location.
        
        Args:
            location: City name or location
            unit: Temperature unit (celsius or fahrenheit)
            
        Returns:
            Weather information
        """
        # This is a mock implementation
        import random
        temperatures = {
            "new york": {"celsius": 20, "fahrenheit": 68},
            "london": {"celsius": 15, "fahrenheit": 59},
            "tokyo": {"celsius": 25, "fahrenheit": 77},
            "sydney": {"celsius": 22, "fahrenheit": 72}
        }
        
        location = location.lower()
        unit = unit.lower()
        
        if location in temperatures:
            temp = temperatures[location][unit]
        else:
            temp = random.randint(15, 30) if unit == "celsius" else random.randint(59, 86)
            
        weather_conditions = ["sunny", "cloudy", "rainy", "windy", "stormy"]
        condition = random.choice(weather_conditions)
        
        return {
            "location": location,
            "temperature": temp,
            "unit": unit,
            "condition": condition,
            "humidity": random.randint(30, 90)
        }
    
    # function call example
    response = agent.chat(
        "帮我查询从2024年1月20日，从北京出发前往上海的航班的票价?",
        tools=None,
        functions=[get_flight_number, get_ticket_price]
    )
    
    # Chat with code execution
    response = agent.chat(
        "Write a Python function to calculate the Fibonacci sequence up to n terms and show me the result for n=10",
        enable_planning=True
    )
    
    # Task planning example
    response = agent.chat(
        "I need to build a simple web scraper that extracts headlines from news websites and stores them in a CSV file. " +
        "Can you help me plan and implement this?",
        enable_planning=True
    )
    
    # Self-checked response
    response = agent.self_check(
        "Explain how quantum computing differs from classical computing and provide a simple code example in Python."
    )
    
    # Error analysis
    errors_analysis = agent.get_errors_analysis()
    print(f"Errors analysis: {json.dumps(errors_analysis, indent=2)}")
    
    
if __name__ == "__main__":
    example_usage()