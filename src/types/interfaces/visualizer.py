from abc import ABC, abstractmethod
from typing import Any

"""
 TODO create visualization!!!
"""


class IVisualizer(ABC):
    @abstractmethod
    def run(self, input_dto: Any) -> Any:
        raise NotImplementedError
