from __future__ import annotations

import yaml

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class ClassNames:
    detection: List[str]
    segmentation: List[str]

    @classmethod
    def load_from(cls, path: str) -> ClassNames:
        file_content = cls._read_yaml_file(path)
        class_names = cls._parse_yaml_file(file_content)

        return class_names

    @staticmethod
    def _parse_yaml_file(content: Dict[Any, Any]) -> ClassNames:
        return ClassNames(**content)
    
    @staticmethod
    def _read_yaml_file(path: str) -> Dict[Any, Any]:
        with open(path, "r") as f:
            return yaml.safe_load(f)


@dataclass
class GroupConfig:
    group_ids: List[int]
    surrogate_id: int
