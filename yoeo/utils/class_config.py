from __future__ import annotations

import yaml

from typing import Dict, List, Any, Optional

from yoeo.utils.dataclasses import ClassNames, GroupConfig


class ClassConfig:
    def __init__(self, content: Dict[Any, Any], class_names: ClassNames):
        self._det_class_names: List[str] = class_names.detection
        self._seg_class_names: List[str] = class_names.segmentation

        self._class_names_to_group: List[str] = content["group_classes"]
        self._group_surrogate_name: Optional[str] = content["surrogate_class"]
        
        self._ids_to_group: Optional[List[int]] = self._compute_group_ids()
        self._grouped_det_class_names: List[str] = self._group_class_names()

    def _compute_group_ids(self) -> Optional[List[int]]:
        """
        Given the list of detection class names and the list of class names that should be grouped into one class,
        compute the ids of the latter classes, i.e. their position in the list of detection class names.

        :return: The ids of all class names that should be grouped into one class if there are any. None otherwise.
        :rtype: Optional[List[int]]
        """
        group_ids = None

        if self._class_names_to_group:
            group_ids = []

            for idx, class_name in enumerate(self._det_class_names):
                if class_name in self._class_names_to_group:
                    group_ids.append(idx)

        return group_ids

    def _group_class_names(self) -> List[str]:
        """
        Given the list of detection class names and the list of class names that should be grouped into one class,
        compute a new list of class names in which all of the latter class names are removed and the surrogate class
        name is inserted at the position of the first class of the classes that should be grouped.

        :return: A list of class names in which all class names that should be grouped are removed and the surrogate
                 class name is inserted as a surrogate for those classes
        :rtype: List[str]
        """

        # Copy the list of detection class names
        grouped_class_names = list(self._det_class_names)

        if self._ids_to_group:
            # Insert the surrogate class name before the first to be grouped class name
            grouped_class_names.insert(self.get_surrogate_id(), self._group_surrogate_name)

            # Remove all to be grouped class names
            for name in self._class_names_to_group:
                grouped_class_names.remove(name)

        return grouped_class_names     
    
    def get_group_config(self) -> Optional[GroupConfig]:
        """ 
        Get the current 'GroupConfig'. 
        
        :return: The current 'GroupConfig' if neither 'self.get_group_ids()' nor 'self.get_surrogate_id()' is
                 'None'. Return 'None' otherwise.
        :rtype: Optional[GroupConfig] 
        """

        group_ids = self.get_group_ids()
        surrogate_id = self.get_surrogate_id()

        if group_ids is None or surrogate_id is None:
            return None
        else:
            return GroupConfig(group_ids, surrogate_id)

    def get_group_class_names(self) -> List[str]:
        """
        Get the class names of the classes that should be grouped together during evaluation

        :return: a list of class names that should be grouped together during evaluation
        :rtype: List[str]
        """
        return self._class_names_to_group

    def get_surrogate_id(self) -> Optional[int]:
        """
        Get the id of the surrogate class in the list of grouped class names. If there are no classes to be grouped, 
        None is returned.

        :return: The id of the surrogate class in the list of grouped class names if there are classes that should be
                 grouped. None otherwise.
        :rtype: Optional[int]
        """
        return None if not self._ids_to_group else self._ids_to_group[0]
        
    def get_grouped_det_class_names(self) -> List[str]:
        """
        Get the grouped list of detection class names.

        :return: The grouped list of detection class names.
        :rtype: List[str]
        """

        return self._grouped_det_class_names

    def get_ungrouped_det_class_names(self) -> List[str]:
        """
        Get the ungrouped list of detection class names.

        :return: The ungrouped list of detection class names.
        :rtype: List[str]
        """

        return self._det_class_names

    def get_seg_class_names(self) -> List[str]:
        """
        Get the list of segmentation class names.

        :return: The list of segmentation class names.
        :rtype: List[str]
        """

        return self._seg_class_names

    def get_group_ids(self) -> Optional[List[int]]:
        """
        Get the (ungrouped) ids of the class names that should be grouped into one class.

        :return: A list of ungrouped ids for the class names that should be grouped into one class if there are any.
                 None otherwise
        :rtype: Optional[List[int]]
        """
        return self._ids_to_group
        
    def get_surrogate_name(self) -> Optional[str]:
        """
        Get the class name of the surrogate class if there are classes that should be grouped into one class. Return
        None otherwise.

        :return: The name of the surrogate class if there are classes that should be grouped into one class. None
                 otherwise.
        :rtype: Optional[List[str]]
        """
                
        return self._group_surrogate_name

    def classes_should_be_grouped(self) -> bool:
        """
        Return true if there are classes that should be grouped into one class. Return false otherwise.

        :return: true if there are classes that should be grouped into on class. False otherwise.
        :rtype: bool
        """
        return self._ids_to_group is not None

    def group(self, labels: List[int]) -> List[int]:
        """
        Group a list of class ids. Given a set of classes that should be grouped X, replace all class ids in X by
        the surrogate id.

        :param labels: list of class ids to group.
        :type labels: List[int]

        :return: grouped list of class ids where 
        :rtype: List[int]
        """
        surrogate_id = self.get_surrogate_id()

        return [label if label not in self._ids_to_group else surrogate_id for label in labels]
    
    @classmethod
    def load_from(cls, path: str, class_names: ClassNames) -> ClassConfig:
        content = cls._read_yaml_file(path)

        return ClassConfig(content, class_names)
    
    @staticmethod
    def _read_yaml_file(path: str) -> Dict[Any, Any]:
        with open(path, "r") as f:
            return yaml.safe_load(f)
