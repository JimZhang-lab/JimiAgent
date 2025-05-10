'''
Author: JimZhang
Date: 2025-05-08 21:44:13
LastEditors: 很拉风的James
LastEditTime: 2025-05-11 05:29:59
FilePath: /Code/Project/JimiAgent/test.py
Description: 

'''

'''
glm-z1-flash
glm-4-flash
b193f25590a05afc5a99678fb7e340d2.3KwMgAhT6WyPgiuO
'''
from component.agent_init import Agent
from component.registry_tools import get_tools, tool_registry
import os
from dotenv import load_dotenv


load_dotenv()

api_key=os.environ['API_KEY']
base_url=os.environ['BASE_URL']
model_name = "glm-z1-flash"
system_prompt = "你是一个聊天大师"
description = "Agent1"
description2 = "Agent2"


@tool_registry({
    "date": "查询航班的日期，格式 YYYY-MM-DD",
    "departure": "出发地",
    "destination": "目的地"
})
def get_flight_number(date: str, departure: str, destination: str):
    """
    根据始发地、目的地和日期，查询对应日期的航班号
    """
    flight_number = {
        "北京": {"上海": "1234", "广州": "8321"},
        "上海": {"北京": "1233", "广州": "8123"},
    }
    return {"flight_number": flight_number[departure][destination]}

@tool_registry({
    "date": "查询票价的日期，格式 YYYY-MM-DD",
    "flight_number": "航班号"
})
def get_ticket_price(date: str, flight_number: str):
    """
    查询某航班在某日的票价
    """
    return {"ticket_price": "1000"}

if __name__ == "__main__":
    config = {
        "api_key": api_key,
        "base_url": base_url,
        "model_name": model_name,
        "system_prompt": "You are a helpful assistant.",
        "description": "My AI assistant",
        "max_history_length": 10
    }
    
    tools = get_tools(['get_flight_number', 'get_ticket_price'])

    agent2 = Agent(**config)
    
    agent2.chat("帮我查询从2024年1月20日，从北京出发前往上海的航班的票价", tools, [get_flight_number, get_ticket_price])
    