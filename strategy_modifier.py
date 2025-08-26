import ast
import importlib.util
import os

def add_new_strategy(file_path, strategy_name, strategy_code):
    """Dynamically add a new strategy function to strategies.py."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found.")

    with open(file_path, 'r') as f:
        code = f.read()
    
    tree = ast.parse(code)
    
    # Check if strategy already exists
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == strategy_name:
            raise ValueError(f"Strategy '{strategy_name}' already exists.")
    
    # Append the new strategy code
    with open(file_path, 'a') as f:
        f.write(f"\n\n# New strategy: {strategy_name}\n{strategy_code}\n")
    
    # Reload to verify
    spec = importlib.util.spec_from_file_location("strategies", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    return f"Strategy '{strategy_name}' added successfully."

def remove_strategy(file_path, strategy_name):
    """Dynamically remove a strategy function from strategies.py."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found.")

    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Find and remove the function definition
    in_func = False
    func_start = -1
    new_lines = []
    for i, line in enumerate(lines):
        if line.startswith(f"def {strategy_name}("):
            in_func = True
            func_start = i
        if in_func and line.strip() == "" and func_start != -1:
            in_func = False
            func_start = -1
            continue  # Skip the function block
        if not in_func:
            new_lines.append(line)
    
    with open(file_path, 'w') as f:
        f.write(''.join(new_lines))
    
    # Reload to verify
    spec = importlib.util.spec_from_file_location("strategies", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if strategy_name in module.strategies:
        raise ValueError(f"Failed to remove '{strategy_name}' from strategies dict.")
    return f"Strategy '{strategy_name}' removed successfully."