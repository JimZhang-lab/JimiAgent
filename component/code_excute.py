'''
Author: JimZhang
Date: 2025-05-09 00:15:38
LastEditors: 很拉风的James
LastEditTime: 2025-05-09 00:23:35
FilePath: /Code/Paper/Agent/component/code_excute.py
Description: 

'''

def execute_code(code_str, function_name, parameters=None):
    """
    Executes a function defined in a string of code.

    Args:
        code_str (str): The code defining the function.
        function_name (str): The name of the function to execute.
        parameters (tuple): The parameters to pass to the function.

    Returns:
        The result of the function execution or an error message.
    """
    try:
        # Prepare the execution environment
        exec_scope = {"__builtins__": {}}
        exec(code_str, exec_scope)

        # Retrieve the function from the execution scope
        func = exec_scope.get(function_name)

        if func is None:
            raise NameError(f"Function '{function_name}' is not defined.")

        # Ensure parameters is a tuple
        if parameters is None:
            parameters = tuple()

        # Execute the function with the provided parameters
        result = func(*parameters)
        return result

    except Exception as e:
        return f"Error: {str(e)}"


# code = """
# def greet(name):
#     return f"Hello, {name}!"
# """

# result = execute_code(code, "greet", ("Alice",))
# print(result)  # Output: Hello, Alice!
