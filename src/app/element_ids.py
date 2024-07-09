"""Element IDs for the app."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class EIdSubstrings:
    """Stores the element ids under the main heading."""

    name: str

    def __post_init__(self):
        """Post init processing."""
        self.container = f"{self.name.replace('_', '-')}--container"
        self.warning = f"{self.name.replace('_', '-')}--warning"

    def add(self, name: str) -> None:
        """Add a new element ID as an attribute."""
        setattr(
            self,
            name.replace("-", "_"),
            f"{self.name.replace('_', '-')}--{name.replace('_', '-')}",
        )


@dataclass
class ElementIds:
    """Element IDs for the app."""

    def add(self, name: str, subnames: str | list[str]) -> None:
        """Add a new element ID as an attribute."""
        setattr(self, name.replace("-", "_"), EIdSubstrings(name))

        if isinstance(subnames, str):
            subnames = [subnames]
        for subname in subnames:
            getattr(self, name).add(subname)

    @classmethod
    def from_json(cls, path: str = "./eids.json") -> ElementIds:
        """Create an ElementIds object from a JSON file."""
        with Path(path.replace("-", "_")).open() as file:
            return cls(**json.load(file))

    def to_json(self, path: str = "./eids.json") -> None:
        """Write the ElementIds object to a JSON file."""
        for key, value in self.__dict__.items():
            if isinstance(value, EIdSubstrings):
                self.__dict__[key] = value.__dict__
        with Path(path.replace("-", "_")).open("w") as file:
            json.dump(self.__dict__, file, indent=4)


__all__ = ["ElementIds"]
