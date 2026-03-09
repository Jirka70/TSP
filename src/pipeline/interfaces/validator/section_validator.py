# pipeline/interfaces/validation/section_validator.py

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.types.dto.raw_section import RawConfigSection


class ISectionValidator(ABC):
    @abstractmethod
    def section_name(self) -> str:
        """
        Name of validated section
        """
        raise NotImplementedError

    @abstractmethod
    def validate(self, raw_section: RawConfigSection) -> bool:
        """
        Zvaliduje jednu sekci configu a vrátí její typovaný DTO/model.

        Při nevalidním vstupu vyhazuje validační výjimku.
        """
        raise NotImplementedError