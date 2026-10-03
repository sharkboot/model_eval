# core/auto_import.py

import importlib
import pkgutil

from core.logger import get_logger

logger = get_logger()


def auto_import(package_name, verbose=False):
    """自动导入包下所有子模块，失败时记录 warning 而非中断。"""
    try:
        package = importlib.import_module(package_name)
    except ImportError as e:
        logger.warning(f"auto_import: 无法导入包 '{package_name}': {e}")
        return

    if not getattr(package, "__path__", None):
        logger.warning(f"auto_import: '{package_name}' 不是包，跳过子模块导入")
        return

    for _, module_name, ispkg in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        try:
            importlib.import_module(module_name)
        except ImportError as e:
            logger.warning(f"auto_import: 导入 {module_name} 失败: {e}")
            if verbose:
                raise