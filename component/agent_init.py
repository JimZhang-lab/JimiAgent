'''
Author: JimZhang
Date: 2025-05-11 05:29:35
LastEditors: 很拉风的James
LastEditTime: 2025-05-11 16:16:57
FilePath: /Code/Project/JimiAgent/component/agent_init.py
Description: 

'''

import shutil
import traceback
import json
import inspect
import ast
import re
from typing import List, Dict, Any, Optional, Union, Callable
import asyncio
import time

from openai import OpenAI
from pydantic import BaseModel, Field


class TaskPlanner:
    """Task planning module to break down complex tasks into manageable steps."""
    
    def __init__(self, agent):
        self.agent = agent
    
    def plan_tasks(self, task_description: str) -> List[Dict]:
        """
        Break down a complex task into smaller subtasks.
        
        Args:
            task_description: Description of the overall task
            
        Returns:
            List of subtasks with their descriptions and status
        """
        prompt = f"""
        I need to break down the following task into smaller, manageable subtasks:
        
        {task_description}
        
        Please create a detailed plan with the following structure:
        1. Analyze what the task requires
        2. Break down the task into 3-7 sequential subtasks
        3. For each subtask, provide:
           - A clear description
           - Any dependencies on previous subtasks
           - Expected output or success criteria
        
        <think>
        Let me analyze this task carefully and break it into logical steps...
        </think>
        
        <answer>
        {{
            "task_plan": [
                {{
                    "id": 1,
                    "name": "subtask_name",
                    "description": "detailed_description",
                    "dependencies": [],
                    "success_criteria": "criteria_to_determine_completion"
                }},
                ...
            ]
        }}
        </answer>
        """
        
        response = self.agent.client.chat.completions.create(
            model=self.agent.model_name,
            messages=[
                {"role": "system", "content": "You are a task planning assistant that breaks down complex tasks into manageable steps."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5
        )
        
        _, answer = self.agent.extract_thinking(response.choices[0].message.content)
        
        try:
            task_plan = json.loads(answer)
            return task_plan["task_plan"]
        except (json.JSONDecodeError, KeyError) as e:
            return [{"id": 1, "name": "execute_full_task", "description": task_description, "dependencies": [], "success_criteria": "Task completed successfully"}]
    
    def execute_plan(self, task_plan: List[Dict], context: Dict = None) -> Dict:
        """
        Execute a task plan step by step.
        
        Args:
            task_plan: List of subtasks to execute
            context: Optional context information for execution
            
        Returns:
            Dictionary containing results of each subtask and overall status
        """
        if context is None:
            context = {}
            
        results = {
            "tasks": {},
            "overall_status": "in_progress",
            "context": context
        }
        
        for task in task_plan:
            task_id = task["id"]
            
            # Check if dependencies are satisfied
            dependencies_met = all(results["tasks"].get(dep, {}).get("status") == "completed" 
                                  for dep in task.get("dependencies", []))
            
            if not dependencies_met:
                results["tasks"][task_id] = {
                    "status": "blocked",
                    "message": "Dependencies not satisfied"
                }
                continue
                
            # Execute the task
            try:
                prompt = f"""
                I need to execute the following subtask as part of a larger plan:
                
                Task: {task['name']}
                Description: {task['description']}
                Success Criteria: {task['success_criteria']}
                
                Context from previous tasks: {json.dumps(results['context'])}
                
                <think>
                Let me carefully execute this task based on the provided description...
                </think>
                
                <answer>
                {{
                    "result": "detailed_result_of_the_task",
                    "status": "completed/failed",
                    "context_updates": {{
                        "key1": "value1",
                        "key2": "value2"
                    }}
                }}
                </answer>
                """
                
                task_result = self.agent.chat(prompt)
                
                try:
                    task_result_json = json.loads(task_result)
                    results["tasks"][task_id] = {
                        "status": task_result_json.get("status", "completed"),
                        "result": task_result_json.get("result", "Task executed"),
                        "context_updates": task_result_json.get("context_updates", {})
                    }
                    
                    # Update context
                    results["context"].update(task_result_json.get("context_updates", {}))
                    
                except json.JSONDecodeError:
                    results["tasks"][task_id] = {
                        "status": "completed",
                        "result": task_result,
                        "context_updates": {}
                    }
                    
            except Exception as e:
                results["tasks"][task_id] = {
                    "status": "failed",
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
        
        # Update overall status
        if any(task.get("status") == "failed" for task in results["tasks"].values()):
            results["overall_status"] = "failed"
        elif all(task.get("status") == "completed" for task in results["tasks"].values()):
            results["overall_status"] = "completed"
        else:
            results["overall_status"] = "partial"
            
        return results


class CodeExecutor:
    """Enhanced code execution module with safety features and debugging capabilities."""
    
    def __init__(self, agent, safe_mode=True):
        self.agent = agent
        self.safe_mode = safe_mode
        self.permitted_modules = {
            'json', 'math', 're', 'datetime', 'time', 'random', 'collections',
            'itertools', 'functools', 'operator', 'string', 'copy', 'uuid',
            'hashlib', 'base64', 'urllib.parse', 'html', 'xml.etree.ElementTree',
            'csv', 'io', 'zipfile', 'gzip', 'bz2', 'lzma', 'struct', 'pickle'
        }
        self.execution_history = []
        
    def analyze_code_safety(self, code_str: str) -> Dict:
        """
        Analyzes code for potential security issues.
        
        Args:
            code_str: The code to analyze
            
        Returns:
            Dictionary with safety analysis
        """
        if not self.safe_mode:
            return {"safe": True, "warnings": []}
            
        try:
            tree = ast.parse(code_str)
            
            # Check for imports
            suspicious_imports = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    module = getattr(node, 'module', None)
                    
                    # Handle direct imports
                    if isinstance(node, ast.Import):
                        for name in node.names:
                            if name.name.split('.')[0] not in self.permitted_modules:
                                suspicious_imports.append(name.name)
                    
                    # Handle from X import Y
                    elif module and module.split('.')[0] not in self.permitted_modules:
                        suspicious_imports.append(module)
            
            # Check for dangerous functions
            dangerous_calls = []
            dangerous_functions = {'eval', 'exec', 'compile', 'os.system', 'subprocess.run', 
                                   'subprocess.call', 'subprocess.Popen', '__import__', 
                                   'open', 'file', 'globals', 'locals'}
                                   
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id in dangerous_functions:
                        dangerous_calls.append(node.func.id)
                    elif isinstance(node.func, ast.Attribute):
                        call_name = f"{self._get_attribute_name(node.func)}"
                        if call_name in dangerous_functions:
                            dangerous_calls.append(call_name)
            
            is_safe = not (suspicious_imports or dangerous_calls)
            warnings = []
            
            if suspicious_imports:
                warnings.append(f"Suspicious imports detected: {', '.join(suspicious_imports)}")
            
            if dangerous_calls:
                warnings.append(f"Potentially dangerous function calls: {', '.join(dangerous_calls)}")
                
            return {
                "safe": is_safe,
                "warnings": warnings
            }
            
        except SyntaxError as e:
            return {
                "safe": False,
                "warnings": [f"Syntax error in code: {str(e)}"]
            }
    
    def _get_attribute_name(self, node):
        """Helper to get the full attribute name."""
        if isinstance(node, ast.Attribute):
            return f"{self._get_attribute_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Name):
            return node.id
        return ""
        
    def execute_code(self, code_str: str, function_name: str = None, parameters: list = None) -> Dict:
        """
        Executes Python code with safety checks and captures detailed results.
        
        Args:
            code_str: The code to execute
            function_name: Optional name of function to call
            parameters: Optional parameters to pass to the function
            
        Returns:
            Dictionary with execution results and metadata
        """
        start_time = time.time()
        
        # Safety analysis
        safety_analysis = self.analyze_code_safety(code_str)
        
        if not safety_analysis["safe"] and self.safe_mode:
            result = {
                "status": "error",
                "result": None,
                "error": "Code safety check failed",
                "warnings": safety_analysis["warnings"],
                "execution_time": 0
            }
            self.execution_history.append(result)
            return result
            
        # Prepare execution environment
        local_vars = {}
        global_vars = {"__builtins__": {}}
        
        for module_name in self.permitted_modules:
            try:
                if '.' in module_name:
                    # For submodules like xml.etree.ElementTree
                    parts = module_name.split('.')
                    main_module = __import__(parts[0])
                    current = main_module
                    for part in parts[1:]:
                        current = getattr(current, part)
                    global_vars[parts[-1]] = current
                else:
                    # For top-level modules
                    global_vars[module_name] = __import__(module_name)
            except ImportError:
                pass
                
        # Execute code
        try:
            # Execute the provided code
            exec(code_str, global_vars, local_vars)
            
            # If function_name is provided, call the specified function
            if function_name:
                if function_name not in local_vars:
                    raise NameError(f"Function '{function_name}' is not defined in the provided code.")
                    
                func = local_vars[function_name]
                if not callable(func):
                    raise TypeError(f"'{function_name}' is not callable.")
                    
                if parameters is None:
                    parameters = []
                    
                func_result = func(*parameters)
                result = {
                    "status": "success",
                    "result": func_result,
                    "error": None,
                    "warnings": safety_analysis["warnings"],
                    "execution_time": time.time() - start_time
                }
            else:
                # If no function specified, return the local variables
                result = {
                    "status": "success",
                    "result": {k: v for k, v in local_vars.items() if not k.startswith('_')},
                    "error": None,
                    "warnings": safety_analysis["warnings"],
                    "execution_time": time.time() - start_time
                }
                
        except Exception as e:
            result = {
                "status": "error",
                "result": None,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "warnings": safety_analysis["warnings"],
                "execution_time": time.time() - start_time
            }
            
        self.execution_history.append(result)
        return result
    
    def debug_code(self, code_str: str, error_message: str = None) -> Dict:
        """
        Debug code with LLM assistance.
        
        Args:
            code_str: Code with potential issues
            error_message: Optional error message from execution
            
        Returns:
            Dictionary with debugging results
        """
        prompt = f"""
        I need to debug the following Python code:
        
        ```python
        {code_str}
        ```
        
        {f"The code produced this error: {error_message}" if error_message else "The code needs to be improved and debugged."}
        
        <think>
        Let me analyze this code carefully to find bugs and suggest improvements...
        
        1. Looking for syntax errors
        2. Checking logic issues
        3. Identifying potential edge cases
        4. Finding best practices violations
        </think>
        
        <answer>
        {{
            "issues": [
                {{
                    "type": "error/warning/improvement",
                    "description": "Description of the issue",
                    "line": "relevant line number or range",
                    "suggestion": "Suggested fix"
                }}
            ],
            "fixed_code": "Complete fixed version of the code",
            "explanation": "Explanation of changes made"
        }}
        </answer>
        """
        
        response = self.agent.client.chat.completions.create(
            model=self.agent.model_name,
            messages=[
                {"role": "system", "content": "You are an expert Python developer focused on debugging and improving code."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        _, answer = self.agent.extract_thinking(response.choices[0].message.content)
        
        try:
            debug_result = json.loads(answer)
            return debug_result
        except json.JSONDecodeError:
            return {
                "issues": [{"type": "error", "description": "Failed to debug code", "suggestion": "Try manual debugging"}],
                "fixed_code": code_str,
                "explanation": "The debugging system encountered an error parsing the LLM response."
            }


class SelfDebugger:
    """Module for self-debugging and error recovery capabilities."""
    
    def __init__(self, agent):
        self.agent = agent
        self.error_log = []
        
    def log_error(self, error_type: str, error_message: str, context: Dict = None):
        """
        Log an error for later analysis.
        
        Args:
            error_type: Type of error
            error_message: Error message
            context: Additional context about the error
        """
        error_entry = {
            "timestamp": time.time(),
            "error_type": error_type,
            "error_message": error_message,
            "context": context or {}
        }
        self.error_log.append(error_entry)
        
    def analyze_errors(self) -> Dict:
        """
        Analyze error patterns to identify recurring issues.
        
        Returns:
            Analysis of error patterns
        """
        if not self.error_log:
            return {"error_patterns": [], "suggestions": []}
            
        prompt = f"""
        I need to analyze the following error log to identify patterns and suggest improvements:
        
        {json.dumps(self.error_log, indent=2)}
        
        <think>
        Let me analyze these errors to find patterns and root causes...
        
        1. Grouping similar errors
        2. Identifying frequent error types
        3. Looking for common contexts where errors occur
        4. Determining if errors are related to specific inputs or operations
        </think>
        
        <answer>
        {{
            "error_patterns": [
                {{
                    "pattern": "Description of error pattern",
                    "frequency": 5,
                    "examples": ["error_examples"],
                    "likely_cause": "Probable root cause"
                }}
            ],
            "suggestions": [
                {{
                    "target": "area_to_improve",
                    "suggestion": "detailed_suggestion",
                    "priority": "high/medium/low"
                }}
            ]
        }}
        </answer>
        """
        
        response = self.agent.client.chat.completions.create(
            model=self.agent.model_name,
            messages=[
                {"role": "system", "content": "You are an expert system analyst specialized in error pattern recognition and debugging."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        _, answer = self.agent.extract_thinking(response.choices[0].message.content)
        
        try:
            analysis = json.loads(answer)
            return analysis
        except json.JSONDecodeError:
            return {"error_patterns": [], "suggestions": [{"target": "error_analysis", "suggestion": "Manual review needed", "priority": "high"}]}
    
    def recover_from_error(self, error_message: str, failed_operation: str, context: Dict = None) -> Dict:
        """
        Attempt to recover from an error and suggest alternatives.
        
        Args:
            error_message: The error message
            failed_operation: Description of what failed
            context: Additional context information
            
        Returns:
            Recovery suggestions and action plan
        """
        context_str = json.dumps(context) if context else "{}"
        
        prompt = f"""
        I encountered an error and need to recover:
        
        Failed operation: {failed_operation}
        Error message: {error_message}
        Context: {context_str}
        
        <think>
        Let me analyze this error and determine the best recovery strategy...
        
        1. Understanding what caused the error
        2. Identifying alternative approaches
        3. Determining if we can proceed with partial results
        4. Designing a recovery plan
        </think>
        
        <answer>
        {{
            "error_analysis": "Analysis of what went wrong",
            "can_recover": true/false,
            "recovery_plan": [
                "Step 1 of recovery",
                "Step 2 of recovery"
            ],
            "alternative_approaches": [
                {{
                    "description": "Alternative approach",
                    "pros": ["advantage1", "advantage2"],
                    "cons": ["disadvantage1", "disadvantage2"]
                }}
            ]
        }}
        </answer>
        """
        
        response = self.agent.client.chat.completions.create(
            model=self.agent.model_name,
            messages=[
                {"role": "system", "content": "You are an expert troubleshooter specialized in error recovery and alternative solution design."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5
        )
        
        _, answer = self.agent.extract_thinking(response.choices[0].message.content)
        
        try:
            recovery_plan = json.loads(answer)
            return recovery_plan
        except json.JSONDecodeError:
            return {
                "error_analysis": "Could not parse recovery plan",
                "can_recover": False,
                "recovery_plan": ["Restart the operation with modified parameters"],
                "alternative_approaches": [{"description": "Manual intervention required", "pros": ["Human oversight"], "cons": ["Time-consuming"]}]
            }


class Function(BaseModel):
    """Class for defining a callable function with signature and documentation."""
    name: str
    description: str
    parameters: Dict[str, Any]
    
    def to_tool_format(self) -> Dict:
        """Convert the function to OpenAI tool format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": list(self.parameters.keys())
                }
            }
        }


class Agent:
    """
    Enhanced AI agent with code execution, task planning, and self-debugging capabilities.
    """
    
    def __init__(self, 
                 api_key: str, 
                 model_name: str = "gpt-3.5-turbo", 
                 base_url: str = "https://api.openai.com", 
                 system_prompt: str = "You are a helpful assistant.", 
                 description: str = "My AI assistant", 
                 max_history_length: int = 10,
                 safe_mode: bool = True,
                 *args, **kwargs):
        """
        Initialize the Agent with enhanced capabilities.

        Args:
            api_key: OpenAI API key
            model_name: Model to use for agent
            base_url: API base URL
            system_prompt: System prompt for the agent
            description: Description of the agent
            max_history_length: Maximum conversation history length
            safe_mode: Whether to run code execution in safe mode
            args: Additional positional arguments
            kwargs: Additional keyword arguments
        """
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url
        self.system_prompt = system_prompt  
        self.description = description
        self.max_history_length = max_history_length
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        
        self.code_executor = CodeExecutor(self, safe_mode=safe_mode)
        self.task_planner = TaskPlanner(self)
        self.self_debugger = SelfDebugger(self)
        
        self.history = []
        self.registered_functions = {}
        self.args = args
        self.kwargs = kwargs
        
    def register_function(self, func: Callable = None, name: str = None, description: str = None) -> None:
        """
        Register a Python function to be available as a tool.
        
        Args:
            func: Function to register
            name: Optional name override for the function
            description: Optional description override
        """
        if func is None:
            # Used as decorator
            def decorator(f):
                self.register_function(f, name, description)
                return f
            return decorator
            
        func_name = name or func.__name__
        func_doc = description or inspect.getdoc(func) or f"Function {func_name}"
        
        # Get function signature
        sig = inspect.signature(func)
        parameters = {}
        
        for param_name, param in sig.parameters.items():
            if param.annotation != inspect.Parameter.empty:
                if hasattr(param.annotation, "__origin__") and param.annotation.__origin__ is Union:
                    # Handle Union types
                    types = [t.__name__ for t in param.annotation.__args__ if t is not type(None)]
                    param_type = "string" if "str" in types else "number" if set(types) & {"int", "float"} else "object"
                else:
                    # Handle simple types
                    type_name = param.annotation.__name__
                    param_type = "string" if type_name == "str" else "number" if type_name in ["int", "float"] else "object"
            else:
                # Default to string if no type annotation
                param_type = "string"
                
            parameters[param_name] = {
                "type": param_type,
                "description": f"Parameter {param_name}"
            }
            
            # Add default value if available
            if param.default != inspect.Parameter.empty:
                parameters[param_name]["default"] = param.default
                
        function_def = Function(
            name=func_name,
            description=func_doc,
            parameters=parameters
        )
        
        self.registered_functions[func_name] = {
            "function": func,
            "definition": function_def
        }
    
    def _prepare_message(self, prompt: str) -> List[Dict]:
        """
        Prepares the message list for the OpenAI API request.
        
        Args:
            prompt: The user's input
            
        Returns:
            List of message dictionaries
        """
        message = [{"role": "system", "content": self.system_prompt}]
        if self.history:
            message.extend(self.history)
        message.append({"role": "user", "content": prompt})
        return message

    def _update_history(self, role: str, content: Union[str, Dict]) -> None:
        """
        Updates the conversation history.
        
        Args:
            role: Role of the message sender
            content: Content of the message
        """
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.max_history_length:
            self.history.pop(0)

    def clear_history(self) -> None:
        """Clears the conversation history."""
        self.history = []

    def extract_thinking(self, llm_output: str) -> tuple:
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
        
    # def extract_thinking(self, llm_output: str) -> tuple:
    #     """
    #     Extracts the thinking process and final answer from LLM output.
        
    #     Args:
    #         llm_output: Raw output from the LLM
            
    #     Returns:
    #         Tuple of (thinking, answer)
    #     """
    #     thinking = ""
    #     answer = llm_output.strip()
        
    #     # Extract thinking section
    #     think_match = re.search(r'<think>(.*?)</think>', llm_output, re.DOTALL)
    #     if think_match:
    #         thinking = think_match.group(1).strip()
            
    #     # Extract answer section
    #     answer_match = re.search(r'<answer>(.*?)</answer>', llm_output, re.DOTALL)
    #     if answer_match:
    #         answer = answer_match.group(1).strip()
            
    #     return thinking, answer

    def execute_python_code(self, code: str, function_name: str = None, parameters: list = None) -> Dict:
        """
        Executes Python code with error handling and debugging assistance.
        
        Args:
            code: Python code to execute
            function_name: Optional function to call after execution
            parameters: Parameters to pass to the function
            
        Returns:
            Result of execution
        """
        result = self.code_executor.execute_code(code, function_name, parameters)
        
        if result["status"] == "error":
            self.self_debugger.log_error("code_execution", result["error"], {"code": code})
            
            # Attempt to debug and fix the code
            debug_result = self.code_executor.debug_code(code, result["error"])
            
            if debug_result.get("fixed_code"):
                # Try executing the fixed code
                result = self.code_executor.execute_code(
                    debug_result["fixed_code"], 
                    function_name, 
                    parameters
                )
                
                if result["status"] == "success":
                    result["debug_info"] = debug_result
                    result["was_fixed"] = True
        
        return result

    def plan_and_execute(self, task_description: str) -> Dict:
        """
        Plan a complex task and execute it step by step.
        
        Args:
            task_description: Description of the task to perform
            
        Returns:
            Result of task execution
        """
        try:
            # Generate task plan
            task_plan = self.task_planner.plan_tasks(task_description)
            
            # Execute the plan
            execution_result = self.task_planner.execute_plan(task_plan)
            
            return {
                "status": "success",
                "task_plan": task_plan,
                "execution_result": execution_result
            }
            
        except Exception as e:
            error_msg = str(e)
            traceback_info = traceback.format_exc()
            
            self.self_debugger.log_error(
                "task_execution", 
                error_msg, 
                {"task": task_description, "traceback": traceback_info}
            )
            
            # Try to recover
            recovery_plan = self.self_debugger.recover_from_error(
                error_msg,
                f"Failed to plan and execute task: {task_description}",
                {"traceback": traceback_info}
            )
            
            return {
                "status": "error",
                "error": error_msg,
                "traceback": traceback_info,
                "recovery_plan": recovery_plan
            }

    def get_function_schema(self, functions: List[Callable] = None) -> List[Dict]:
        """
        Get OpenAI tool schema for registered functions.
        
        Args:
            functions: Optional list of additional functions to include
            
        Returns:
            List of function schemas in OpenAI tool format
        """
        tools = []
        
        # Add registered functions
        for func_info in self.registered_functions.values():
            tools.append(func_info["definition"].to_tool_format())
            
        # Add code execution tool
        code_execution_tool = {
            "type": "function",
            "function": {
                "name": "execute_python_code",
                "description": "Execute Python code and return the result",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python code to execute"
                        },
                        "function_name": {
                            "type": "string",
                            "description": "Optional function to call after executing the code"
                        },
                        "parameters": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "Parameters to pass to the function"
                        }
                    },
                    "required": ["code"]
                }
            }
        }
        tools.append(code_execution_tool)
        
        # Add task planning tool
        task_planning_tool = {
            "type": "function",
            "function": {
                "name": "plan_and_execute",
                "description": "Plan a complex task and execute it step by step",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_description": {
                            "type": "string",
                            "description": "Description of the task to plan and execute"
                        }
                    },
                    "required": ["task_description"]
                }
            }
        }
        tools.append(task_planning_tool)
        
        return tools

    def is_function_call(self, response) -> bool:
        """
        Check if the response contains a function call.
        
        Args:
            response: API response object
            
        Returns:
            Boolean indicating if it's a function call
        """
        # Check for standard tool/function calls
        if hasattr(response.choices[0].message, 'tool_calls') and response.choices[0].message.tool_calls:
            return True
            
        # Check for reasoning model function calls embedded in content
        if hasattr(response.choices[0].message, 'content') and response.choices[0].message.content:
            try:
                _, answer = self.extract_thinking(response.choices[0].message.content)
                answer_json = json.loads(answer)
                if 'delta' in answer_json and 'tool_calls' in answer_json['delta'] and answer_json['delta']['tool_calls']:
                    return True
            except (json.JSONDecodeError, KeyError, AttributeError):
                pass
                
        return False
        
    def get_reasoning_model_function(self, response):
        """
        Extracts the reasoning model function from the response.
        
        Args:
            response: The response from the LLM
            
        Returns:
            The reasoning model function call or None
        """
        try:
            _, answer = self.extract_thinking(response.choices[0].message.content)
            answer_json = json.loads(answer)
            if 'delta' in answer_json and 'tool_calls' in answer_json['delta'] and answer_json['delta']['tool_calls'][0]:
                return answer_json['delta']['tool_calls'][0]
        except (json.JSONDecodeError, KeyError, AttributeError, IndexError):
            pass
        return None
        
    def parse_function_call(self, model_response, messages, tools=None, functions=None):
        """
        Handle function calling, execute the functions, and process results.
        
        Args:
            model_response: Response from the model
            messages: Current message history
            tools: Available tools
            functions: List of callable functions
            
        Returns:
            Content of the assistant's response after function execution
        """
        if functions is None:
            functions = []
            
        tool_call = None
        function_result = {}

        # Process response from reasoning models
        if model_response.choices[0].message.content:
            tool_call = self.get_reasoning_model_function(model_response)
            
        # Process response from standard models
        elif hasattr(model_response.choices[0].message, 'tool_calls') and model_response.choices[0].message.tool_calls:
            tool_call = model_response.choices[0].message.tool_calls[0].model_dump()
            
        # Execute the function if found
        if tool_call:
            function_name = tool_call['function']['name']
            args_str = tool_call['function']['arguments']
            
            try:
                args = json.loads(args_str)
                
                # Handle built-in agent functions
                if function_name == "execute_python_code":
                    function_result = self.execute_python_code(
                        args["code"], 
                        args.get("function_name"), 
                        args.get("parameters")
                    )
                elif function_name == "plan_and_execute":
                    function_result = self.plan_and_execute(args["task_description"])
                else:
                    # Handle registered functions
                    if function_name in self.registered_functions:
                        func = self.registered_functions[function_name]["function"]
                        function_result = func(**args)
                    else:
                        # Try to find the function in the provided list
                        for func in functions:
                            if func.__name__ == function_name:
                                function_result = func(**args)
                                break
                        else:
                            function_result = f"Error: Function '{function_name}' not found"
                            
            except Exception as e:
                error_msg = str(e)
                tb = traceback.format_exc()
                function_result = f"Error: {error_msg}\n{tb}"
                
                # Log the error
                self.self_debugger.log_error(
                    "function_execution", 
                    error_msg, 
                    {"function": function_name, "arguments": args_str, "traceback": tb}
                )
            
            # Add function result to messages
            messages.append({
                "role": "tool",
                "content": json.dumps(function_result),
                "tool_call_id": tool_call['id']
            })

            # Call the model again with the function result
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=tools,
                **self.kwargs
            )
            
            # Update message history
            messages.append(response.choices[0].message.model_dump())
            
            # Handle nested function calls recursively
            if self.is_function_call(response):
                return self.parse_function_call(response, messages, tools, functions)
            else:
                return response.choices[0].message.content
        else:
            # No function call, return content directly
            return model_response.choices[0].message.content
            
    def chat(self, prompt, tools=None, functions=None, enable_planning=False, enable_self_debug=True, *args, **kwargs):
        """
        Enhanced chat method with code execution, task planning, and self-debugging capabilities.

        Args:
            prompt: User input
            tools: List of tools to make available
            functions: List of callable functions
            enable_planning: Whether to enable automatic task planning
            enable_self_debug: Whether to enable self-debugging
            args: Additional positional arguments
            kwargs: Additional keyword arguments
            
        Returns:
            The assistant's response
        """
        # Basic task planning for complex tasks
        if enable_planning and any(keyword in prompt.lower() for keyword in 
                                  ["complex", "step by step", "multistep", "multi-step", "plan"]):
            planning_result = self.plan_and_execute(prompt)
            if planning_result["status"] == "success":
                self._update_history("user", prompt)
                plan_response = f"I've planned and executed this task for you:\n\n{json.dumps(planning_result, indent=2)}"
                self._update_history("assistant", plan_response)
                
                # Display output
                try:
                    width = shutil.get_terminal_size().columns
                except:
                    width = 100  # Default width
                print("-" * width)
                print(f"{self.description}\n{plan_response}\n")
                return plan_response
        
        # Prepare messages and tools
        messages = self._prepare_message(prompt)
        self._update_history("user", prompt)
        
        # Register built-in tools if not provided
        if tools is None:
            tools = self.get_function_schema(functions)
            
        # Process the request
        try:
            response = self.client.chat.completions.create(
                messages=messages,
                model=self.model_name,
                tools=tools,
                temperature=kwargs.get('temperature', 0.7),
                top_p=kwargs.get('top_p', 0.9),
                **self.kwargs
            )
            
            # Handle function calling if present
            if self.is_function_call(response):
                thinking, answer = self.extract_thinking(self.parse_function_call(response, messages, tools, functions))
            else:
                # Extract thinking and answer
                thinking, answer = self.extract_thinking(response.choices[0].message.content)
                
            # Update conversation history
            self._update_history("assistant", answer)
            
        except Exception as e:
            error_msg = str(e)
            tb = traceback.format_exc()
            
            if enable_self_debug:
                # Log the error and try to recover
                self.self_debugger.log_error("api_error", error_msg, {"traceback": tb})
                recovery_info = self.self_debugger.recover_from_error(
                    error_msg, 
                    "API request failed", 
                    {"prompt": prompt}
                )
                
                error_response = f"Sorry, I encountered an issue: {error_msg}\n\nI'll try to recover..."
                
                # Try a simpler request if possible
                if recovery_info.get("can_recover", False):
                    try:
                        # Simpler request without tools
                        simple_response = self.client.chat.completions.create(
                            messages=messages,
                            model=self.model_name,
                            temperature=0.5,
                        )
                        
                        _, simple_answer = self.extract_thinking(simple_response.choices[0].message.content)
                        answer = f"I had to use a simpler approach due to an error.\n\n{simple_answer}"
                        self._update_history("assistant", answer)
                    except:
                        answer = f"{error_response}\n\nUnfortunately, I couldn't recover automatically. Error details: {error_msg}"
                        self._update_history("assistant", answer)
                else:
                    answer = f"{error_response}\n\nUnfortunately, recovery wasn't possible. Error details: {error_msg}"
                    self._update_history("assistant", answer)
            else:
                answer = f"Sorry, I encountered an issue while processing your request. API error: {error_msg}"
                self._update_history("assistant", answer)
        
        # Display output with dynamic width
        try:
            width = shutil.get_terminal_size().columns
        except:
            width = 100  # Default width
        print("-" * width)
        print(f"{self.description}\n{answer}\n")
        
        return answer
        
    def self_check(self, prompt, tools=None, functions=None, *args, **kwargs):
        """
        Run self-checking version of chat that verifies its own outputs.
        
        Args:
            prompt: User input
            tools: List of tools to make available
            functions: List of callable functions
            args: Additional positional arguments
            kwargs: Additional keyword arguments
            
        Returns:
            The verified response
        """
        # Generate initial response
        initial_response = self.chat(prompt, tools, functions, *args, **kwargs)
        
        # Self-verification prompt
        verification_prompt = f"""
        I need to verify the correctness and quality of my previous response.
        
        User query: {prompt}
        
        My response: {initial_response}
        
        <think>
        Let me check my response for:
        1. Factual accuracy
        2. Completeness - did I address all parts of the query?
        3. Clarity and coherence
        4. Code correctness (if code was provided)
        5. Any logical errors or inconsistencies
        </think>
        
        <answer>
        {{
            "is_correct": true/false,
            "issues": [
                {{
                    "issue_type": "factual_error/incompleteness/logic_error/code_bug/other",
                    "description": "Description of the issue",
                    "correction": "Corrected information"
                }}
            ],
            "improved_response": "The improved response if issues were found, otherwise null"
        }}
        </answer>
        """
        
        # Run verification
        verification_result = self.chat(verification_prompt, tools=None, functions=None, temperature=0.2)
        
        try:
            # Process verification result
            verification_data = json.loads(verification_result)
            
            if verification_data["is_correct"]:
                return initial_response
            else:
                # Return improved response if issues were found
                if verification_data.get("improved_response"):
                    corrected_response = verification_data["improved_response"]
                    
                    # Update history with the corrected response
                    self.history.pop()  # Remove the verification response
                    self.history.pop()  # Remove the verification prompt
                    self.history.pop()  # Remove the initial response
                    
                    self._update_history("assistant", corrected_response)
                    
                    # Display output
                    try:
                        width = shutil.get_terminal_size().columns
                    except:
                        width = 100  # Default width
                    print("-" * width)
                    print(f"{self.description} (Self-corrected)\n{corrected_response}\n")
                    
                    return corrected_response
                else:
                    return initial_response
        except (json.JSONDecodeError, KeyError):
            # If verification parsing fails, return the original response
            return initial_response
    
    def get_errors_analysis(self):
        """
        Get an analysis of errors encountered by the agent.
        
        Returns:
            Analysis of error patterns and suggestions
        """
        return self.self_debugger.analyze_errors()
        
    async def achat(self, prompt, tools=None, functions=None, timeout=30, *args, **kwargs):
        """
        Asynchronous version of chat with timeout handling.
        
        Args:
            prompt: User input
            tools: List of tools to make available
            functions: List of callable functions
            timeout: Maximum time to wait for response (in seconds)
            args: Additional positional arguments
            kwargs: Additional keyword arguments
            
        Returns:
            The assistant's response
        """
        # Define the chat task
        async def chat_task():
            # Create an event loop to run synchronous code
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, 
                lambda: self.chat(prompt, tools, functions, *args, **kwargs)
            )
        
        try:
            # Run with timeout
            return await asyncio.wait_for(chat_task(), timeout=timeout)
        except asyncio.TimeoutError:
            error_msg = f"Response timed out after {timeout} seconds"
            self.self_debugger.log_error("timeout", error_msg, {"prompt": prompt})
            
            # Try to generate a simpler response
            simplified_prompt = f"The previous request timed out. Please provide a brief, simplified response to: {prompt}"
            
            try:
                # Use a basic model call without tools to get quick response
                simple_response = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant. Keep responses very brief."},
                        {"role": "user", "content": simplified_prompt}
                    ],
                    model=self.model_name,
                    max_tokens=150,
                    temperature=0.5,
                )
                
                answer = f"I'm sorry, but your request timed out after {timeout} seconds. Here's a simplified response:\n\n" + simple_response.choices[0].message.content
            except:
                answer = f"I'm sorry, but your request timed out after {timeout} seconds. Please try a simpler query or break your request into smaller parts."
                
            self._update_history("assistant", answer)
            return answer