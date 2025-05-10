'''
Author: JimZhang
Date: 2025-05-08 21:50:41
LastEditors: 很拉风的James
LastEditTime: 2025-05-09 03:14:31
FilePath: /Code/Paper/Agent/component/agent_init.py
Description: 

'''

import shutil
from openai import OpenAI
import traceback
import json


from pydantic import Field, BaseModel
from typing import Optional

class Agent:
    def __init__(self, 
                 api_key, 
                 model_name="gpt-3.5-turbo", 
                 base_url="https://api.openai.com", 
                 system_prompt="You are a helpful assistant.", 
                 description="My AI assistant", 
                 max_history_length=10,
                 *args, **kwargs):
        """
        Initialize the Agent instance.

        :param config: Dictionary containing initialization parameters.
        :param args: Additional positional parameters.
        :param kwargs: Additional keyword parameters.
        """
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url
        self.system_prompt = system_prompt  
        self.description = description
        self.max_history_length = max_history_length
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        
        self.history = []
        self.args = args
        self.kwargs = kwargs

    def _prepare_message(self, prompt):
        """
        Prepares the message list for the OpenAI API request, including history and prompt.
        
        :param prompt: The user's input.
        :return: Message list to send to the API.
        """
        message = [{"role": "system", "content": self.system_prompt}]
        if self.history:
            message.extend(self.history)
        message.append({"role": "user", "content": prompt})
        return message

    def _update_history(self, role, content):
        """
        Updates the conversation history.
        
        :param role: Role of the message sender ("user" or "assistant").
        :param content: The content of the message.
        """
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.max_history_length:
            self.history.pop(0)

    def clear_history(self):
        """Clears the conversation history."""
        self.history = []

    def extract_thinking(self, llm_output):
        """
        Extracts the thinking process and final answer from LLM output.
        
        :param llm_output: Raw output from the LLM.
        :return: (thinking, answer) tuple.
        """
        try:
            tmp = llm_output.split('<think>')[-1]
            thinking = tmp.split('</think>')[0].strip()
            answer = tmp.split('</think>')[-1].split('<answer>')[-1].split('</answer>')[0].strip()
            return thinking, answer
        except IndexError:
            return "", llm_output.strip()

    def execute_code(self, code_str, function_name, parameters=None):
        """
        Executes a function defined in a string of code.
        
        :param code_str: The code that defines the function.
        :param function_name: The function to execute.
        :param parameters: Parameters to pass to the function.
        :return: The result of the function or an error message.
        """
        try:
            exec_scope = {"__builtins__": {}}
            exec(code_str, exec_scope)
            func = exec_scope.get(function_name)
            if func is None:
                raise NameError(f"Function '{function_name}' is not defined.")
            if parameters is None:
                parameters = tuple()
            return func(*parameters)
        except Exception as e:
            return f"Error: {str(e)}\n{traceback.format_exc()}"
    
    def get_reasoing_model_function(self, response):
        """
        Extracts the reasoning model function from the response.
        
        :param response: The response from the LLM.
        :return: The reasoning model function.
        """
        try:
            _, answer = self.extract_thinking(response.choices[0].message.content)
            answer = json.loads(answer)
            return answer['delta']['tool_calls'][0]
        except Exception as e:
            # print(f"Error extracting reasoning model function: {e}")
            return None
        
    def is_function_call(self, response):
        """
            Checks if the response contains a function call.
            
            :param response: The response from the LLM.
            :return: True if it's a function call, False otherwise.
        """
        if hasattr(response.choices[0].message, 'tool_calls'):
            if response.choices[0].message.tool_calls:
                return True
            elif self.get_reasoing_model_function(response):
                return True
        return False
            

    def chat(self, prompt, tools=None, function=[],*args, **kwargs):
        """
        Conducts a conversation with the user.

        :param prompt: The user's input.
        :param tools: Tools to use for the conversation (optional).
        :param args: Additional positional parameters.
        :param kwargs: Additional keyword parameters.
        :return: The assistant's response.
        """
        message = self._prepare_message(prompt)
        self._update_history("user", prompt)

        try:
            response = self.client.chat.completions.create(
                messages=message,
                model=self.model_name,
                top_p=kwargs.get('top_p', 0.7),
                temperature=kwargs.get('temperature', 0.9),
                tools=tools,
                **self.kwargs
            )
        except Exception as e:
            print(f"API error: {e}")
            return "Sorry, I encountered an issue while processing your request."
        # print(response.choices[0].message)
        # print(response.choices[0].message.model_dump())
        # message.append(response.choices[0].message.model_dump())
        
        if self.is_function_call(response):
            _, answer = self.extract_thinking(self.parse_function_call(response, message, tools, function))
        
        else:
            thinking, answer = self.extract_thinking(response.choices[0].message.content)
        self._update_history("assistant", answer)
        
        # Display output with dynamic width
        try:
            width = shutil.get_terminal_size().columns
        except:
            width = 100  # Default width
        print("-" * width)
        print(f"{self.description}\n{answer}\n")
        # print(self.history)
        return answer

    def parse_function_call(self, model_response, messages, tools=None, function=[]):
        # 处理函数调用结果，根据模型返回参数，调用对应的函数。
        # 调用函数返回结果后构造tool message，再次调用模型，将函数结果输入模型
        # 模型会将函数调用结果以自然语言格式返回给用户。
        
        tool_call = None
        function_result = {}

        # 对于 reasoning 模型
        if model_response.choices[0].message.content:
            _, answer = self.extract_thinking(model_response.choices[0].message.content)
            tool_call = json.loads(answer)
            tool_call = tool_call['delta']['tool_calls'][0]
            # tool_call = tuple(tool_call.items())
            
        # 对于普通模型
        elif model_response.choices[0].message.tool_calls:
            tool_call = model_response.choices[0].message.tool_calls[0].dict()
            # 转为字典
            # tool_call = json.loads(tool_call)
            # print(tool_call)
        # 如果找到tool_call，则处理
        if tool_call:
            args = tool_call['function']['arguments']
            try:
                for func in function:
                    if func.__name__ == tool_call['function']['name']:
                        # 这里可以根据函数名来调用对应的函数
                        # 例如：get_flight_number、get_ticket_price
                        # print(func.__name__)
                        # print(args)
                        function_result = func(**json.loads(args))
            except Exception as e:
                print(f"Error: {e}")
                function_result = f"Error: {str(e)}\n{traceback.format_exc()}"
            
            # 将函数调用结果作为消息追加到tools
            messages.append({
                "role": "tool",
                "content": f"{json.dumps(function_result)}",
                "tool_call_id": tool_call['id']
            })
            # print()
            # print(function_result)
            # print()

            # 用新的消息调用模型
            response = self.client.chat.completions.create(
                model=self.model_name,  # 填写需要调用的模型名称
                messages=messages,
                tools=tools,
            )
            # print(response.choices[0].message)
            
            # self.parse_function_call(response, messages, tools, function)
            # messages.append(response.choices[0].message.model_dump())
        if self.is_function_call(response):
            # 递归调用
            messages.append(response.choices[0].message.model_dump())
            return self.parse_function_call(response, messages, tools, function)
        else:
            return response.choices[0].message.content
            
    def self_check(self, prompt, tools=None, *args, **kwargs):
        pass

