#!./.env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-09-06 04:36:56 (ywatanabe)"
# 2024-09-06 04:33:45

import importlib
import inspect
import os

# Get the current directory
current_dir = os.path.dirname(__file__)

# Iterate through all Python files in the current directory
for filename in os.listdir(current_dir):
    if (
        filename.startswith("_")
        and filename.endswith(".py")
        and not filename.startswith("__")
    ):
        module_name = filename[:-3]  # Remove .py extension
        module = importlib.import_module(f".{module_name}", package=__name__)

        # Import only functions and classes from the module
        for name, obj in inspect.getmembers(module):
            if inspect.isfunction(obj) or inspect.isclass(obj):
                if not name.startswith("_"):
                    globals()[name] = obj

# Clean up temporary variables
del (
    os,
    importlib,
    inspect,
    current_dir,
    filename,
    module_name,
    module,
    name,
    obj,
)

# from ._ import parse_lpath
# from ._ import define_transitional_colors
# from ._ import load_NTdf

# EOF
