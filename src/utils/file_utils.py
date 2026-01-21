import yaml
import importlib


def read_yaml(file_path) -> dict:
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def call_function_by_path(path: str, *args, **kwargs):
    """
    Dynamically call a function by its full path (e.g. 'my_module.my_submodule.my_function').

    Args:
        path (str): The full dotted path to the function.
        *args: Positional arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.

    Returns:
        The result of the function call.
    """
    module_path, func_name = path.rsplit('.', 1)

    module = importlib.import_module(module_path)
    func = getattr(module, func_name)

    return func(*args, **kwargs)
