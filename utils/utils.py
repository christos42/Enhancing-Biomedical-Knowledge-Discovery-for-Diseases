"""JSON and file helpers shared by the pipeline scripts."""

from __future__ import annotations

import json
import os
from typing import Any


def save_json(file: Any, name: str, output_path: str = "") -> None:
    """Write ``file`` as JSON to ``output_path + name``."""
    with open(output_path + name, "w") as outfile:
        json.dump(file, outfile)


def read_json(f_path: str) -> Any:
    """Load and return the JSON content of ``f_path``."""
    f = open(f_path)
    data = json.load(f)
    return data


def find_json_files(path: str) -> list[str]:
    """Return the sorted paths of the ``.json`` files under ``path`` (recursively)."""
    f_path = []
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            if name.endswith(".json"):
                f_path.append(os.path.join(root, name))

    # Sorted, so that listings of different folders line up (os.walk order is
    # filesystem-dependent)
    return sorted(f_path)


def find_csv_files(path: str) -> list[str]:
    """Return the sorted paths of the ``.csv`` files under ``path`` (recursively)."""
    f_path = []
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            if name.endswith(".csv"):
                f_path.append(os.path.join(root, name))

    # Sorted, so that listings of different folders line up (os.walk order is
    # filesystem-dependent)
    return sorted(f_path)


def create_new_folder(path: str) -> None:
    """Create ``path``, with its parents, if it does not exist."""
    exists = os.path.exists(path)
    if not exists:
        os.makedirs(path)


def no_intersection_lists(list1: list[str], list2: list[str]) -> list[str]:
    """Return the items of ``list1`` that are not in ``list2``, keeping their order."""
    # Constant-time membership; a list made this quadratic for large PMID lists
    set2 = set(list2)
    no_inter_list = []
    for item in list1:
        if item not in set2:
            no_inter_list.append(item)

    return no_inter_list
