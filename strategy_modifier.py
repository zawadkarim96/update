import ast
import importlib.util
import os


def add_new_strategy(file_path, strategy_name, strategy_code):
    """Dynamically add a new strategy function to strategies.py."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found.")

    with open(file_path, "r") as f:
        code = f.read()

    tree = ast.parse(code)

    # Check if strategy already exists
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == strategy_name:
            raise ValueError(f"Strategy '{strategy_name}' already exists.")

    # Append the new strategy code
    with open(file_path, "a") as f:
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

    with open(file_path, "r") as f:
        source = f.read()

    tree = ast.parse(source)

    # Remove the function from the AST body
    new_body = []
    removed = False
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == strategy_name:
            removed = True
            continue
        new_body.append(node)

    if not removed:
        raise ValueError(f"Strategy '{strategy_name}' not found.")

    tree.body = new_body

    # Write the updated source back to the file
    new_code = ast.unparse(tree)
    with open(file_path, "w") as f:
        f.write(new_code + "\n")

    # Reload to verify
    spec = importlib.util.spec_from_file_location("strategies", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, "strategies") and strategy_name in module.strategies:
        raise ValueError(f"Failed to remove '{strategy_name}' from strategies dict.")
    return f"Strategy '{strategy_name}' removed successfully."

