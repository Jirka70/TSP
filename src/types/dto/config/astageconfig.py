import importlib

from pydantic import BaseModel


class AStageConfig(BaseModel):
    """
    Abstract class for handling dynamic instantiation based on _target_class attribute
    """

    def __init_subclass__(cls, **kwargs):
        """
        Checks existence of _target_class attribute
        """
        super().__init_subclass__(**kwargs)

        # Check if the child class defined the attribute
        if not hasattr(cls, "_target_class"):
            raise TypeError(f"Configuration class '{cls.__name__}' MUST define '_target_class'")

    def get_instance(self):
        """
        Returns instance of _target_class
        """
        module_path, class_name = self._target_class.rsplit(".", 1)
        module = importlib.import_module(module_path)
        target_class = getattr(module, class_name)

        return target_class()
