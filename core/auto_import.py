# core/auto_import.py

import importlib
import pkgutil


def auto_import(package_name):
    package = importlib.import_module(package_name)

    for _, module_name, _ in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        importlib.import_module(module_name)