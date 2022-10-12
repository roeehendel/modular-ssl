import importlib
import os
import sys
from pathlib import Path
from typing import Callable


class Registry:
    def __init__(self, registered_objects_name: str, module_file: str, module_name: str):
        self.file_root = Path(module_file).parent
        self.registry_name = registered_objects_name
        self.registry = {}
        self.registry_class_names = set()

        setattr(sys.modules[module_name], f"registry", self)

        import_all_modules(self.file_root, module_name)

    # def import_modules(self):
    #     import_all_modules(self.file_root, "ssl_methods")

    def register(self, name: str):
        f"""
        Decorator to register a new {self.registry_name}
        :param name: 
        :return: 
        """

        def register_model_head_cls(cls: Callable[..., Callable]):
            if name in self.registry:
                raise ValueError(f"Cannot register duplicate {self.registry_name} name: {name}")

            if cls.__name__ in self.registry_class_names:
                raise ValueError(f"Cannot register duplicate {self.registry_name} class: {cls.__name__}")
            self.registry[name] = cls
            self.registry_class_names.add(cls.__name__)
            return cls

        return register_model_head_cls

    def get(self, name: str):
        assert name in self.registry, f"Unknown {self.registry_name}: {name}"
        return self.registry[name]

    def keys(self):
        return list(self.registry.keys())


def import_all_modules(root: str, base_module: str) -> None:
    """
    Import all modules in a directory recursively.
    Taken from ClassyVision.
    :param root:
    :param base_module:
    :return:
    """
    for file in os.listdir(root):
        if file.endswith((".py", ".pyc")) and not file.startswith("_"):
            module = file[: file.find(".py")]
            if module not in sys.modules:
                module_name = ".".join([base_module, module])
                importlib.import_module(module_name)
