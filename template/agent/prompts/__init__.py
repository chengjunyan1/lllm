import os
import importlib # Import the library
from lllm.models import Prompt
from lllm.llm import register_prompt

# PROMPT_REGISTRY = {}

# automatically register all prompts in the folder
current_dir = os.path.dirname(__file__)
package_name = __name__ # The package where this __init__.py resides

for file in os.listdir(current_dir):
    if file.endswith('.py') and file not in ['__init__.py', 'basic.py']:
        module_name = file[:-3]
        full_module_path = f'.{module_name}' # Relative import within the package

        # try:
        # Use importlib.import_module for relative import within the package
        module = importlib.import_module(full_module_path, package=package_name)

        # print(f"Processing module: {module.__name__}")
        # print('--------------------------------')

        # Iterate through the attributes of the correctly imported module
        for name, obj in module.__dict__.items():
            # Optional: print to debug what's being found
            # print(f"  Found: {name} (type: {type(obj)})")
            if isinstance(obj, Prompt):
                # print(f"  Registering Prompt: {name}")
                # Adjust path logic if needed - ensure module_name is just the file name part
                obj.path = f'{module_name}/{obj.path}'
                register_prompt(obj)
            # else:
                # print(f"  Skipping: {name}")


        # except ImportError as e:
        #     print(f"Could not import module {full_module_path} from {package_name}: {e}")
        # except Exception as e:
        #      print(f"An error occurred processing module {module_name}: {e}")

