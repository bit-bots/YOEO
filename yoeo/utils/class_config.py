from __future__ import annotations

import yaml

from typing import Dict, List, Any, Optional

from yoeo.utils.dataclasses import ClassNames, SqueezeConfig


class ClassConfig:
    def __init__(self, content: Dict[Any, Any], class_names: ClassNames):
        self._det_class_names: List[str] = class_names.detection
        self._seg_class_names: List[str] = class_names.segmentation

        self._class_names_to_squeeze: List[str] = content["squeeze_classes"]
        self._squeeze_surrogate_name: Optional[str] = content["surrogate_class"]
        
        self._ids_to_squeeze: Optional[List[int]] = self._compute_squeeze_ids()
        self._squeezed_det_class_names: List[str] = self._squeeze_class_names()

    def _compute_squeeze_ids(self) -> Optional[List[int]]:
        """
        Given the list of detection class names and the list of class names that should be squeezed into one class,
        compute the ids of the latter classes, i.e. their position in the list of detection class names.

        :return: The ids of all class names that should be squeezed into one class if there are any. None otherwise.
        :rtype: Optional[List[int]]
        """
        squeeze_ids = None

        if self._class_names_to_squeeze:
            squeeze_ids = []

            for idx, class_name in enumerate(self._det_class_names):
                if class_name in self._class_names_to_squeeze:
                    squeeze_ids.append(idx)

        return squeeze_ids

    def _squeeze_class_names(self) -> List[str]:
        """
        Given the list of detection class names and the list of class names that should be squeezed into one class,
        compute a new list of class names in which all of the latter class names are removed and the surrogate class
        name is inserted at the position of the first class of the classes that should be squeezed.

        :return: A list of class names in which all class names that should be squeezed are removed and the surrogate
                 class name is inserted as a surrogate for those classes
        :rtype: List[str]
        """

        # Copy the list of detection class names
        squeezed_class_names = list(self._det_class_names)

        if self._class_names_to_squeeze:
            # Insert the surrogate class name before the first to be squeezed class name
            squeezed_class_names.insert(self.get_surrogate_id(), self._squeeze_surrogate_name)

            # Remove all to be squeezed class names
            for name in self._class_names_to_squeeze:
                squeezed_class_names.remove(name)

        return squeezed_class_names     
    
    def get_squeeze_config(self) -> Optional[SqueezeConfig]:
        """ 
        Get the current 'SqueezeConfig'. 
        
        :return: The current 'SqueezeConfig' if neither 'self.get_squeeze_ids()' nor 'self.get_surrogate_id()' is
                 'None'. Return 'None' otherwise.
        :rtype: Optional[SqueezeConfig] 
        """

        squeeze_ids = self.get_squeeze_ids()
        surrogate_id = self.get_surrogate_id()

        if squeeze_ids is None or surrogate_id is None:
            return None
        else:
            return SqueezeConfig(squeeze_ids, surrogate_id)

    def get_squeeze_class_names(self) -> List[str]:
        return self._class_names_to_squeeze

    def get_surrogate_id(self) -> Optional[int]:
        """
        Get the id of the surrogate class in the list of squeezed class names. If there are no classes to squeezed, 
        None is returned.

        :return: The id of the surrogate class in the list of squeezed class names if there are classes that should be
                 squeezed. None otherwise.
        :rtype: Optional[int]
        """
        return None if not self._ids_to_squeeze else self._ids_to_squeeze[0]
        
    def get_det_class_names(self) -> List[str]:
        """
        Get the (squeezed) list of detection class names.

        :return: The (squeezed) list of detection class names.
        :rtype: List[str]
        """

        return self._squeezed_det_class_names

    def get_seg_class_names(self) -> List[str]:
        """
        Get the list of segmentation class names.

        :return: The list of segmentation class names.
        :rtype: List[str]
        """

        return self._seg_class_names

    def get_squeeze_ids(self) -> Optional[List[int]]:
        """
        Get the (unsqueezed) ids of the class names that should be squeezed into one class.

        :return: A list of unsqueezed ids for the class names that should be squeezed into one class if there are any.
                 None otherwise
        :rtype: Optional[List[int]]
        """
        return self._ids_to_squeeze
        
    def get_surrogate_name(self) -> Optional[str]:
        """
        Get the class name of the surrogate class if there are classes that should be squeezed into one class. Return
        None otherwise.

        :return: The name of the surrogate class if there are classes that should be squeezed into one class. None
                 otherwise.
        :rtype: Optional[List[str]]
        """
                
        return self._squeeze_surrogate_name

    def classes_should_be_squeezed(self) -> bool:
        """
        Return true if there are classes that should be squeezed into one class. Return false otherwise.

        :return: true if there are classes that should be squeezed into on class. False otherwise.
        :rtype: bool
        """
        return self._ids_to_squeeze is not None

    def squeeze(self, labels: List[int]) -> List[int]:
        surrogate_id = self.get_surrogate_id()

        return [label if label not in self._ids_to_squeeze else surrogate_id for label in labels]

    @classmethod
    def load_from(cls, path: str, class_names: ClassNames) -> ClassConfig:
        content = cls._read_yaml_file(path)

        return ClassConfig(content, class_names)
    
    @staticmethod
    def _read_yaml_file(path: str) -> Dict[Any, Any]:
        with open(path, "r") as f:
            return yaml.safe_load(f)
