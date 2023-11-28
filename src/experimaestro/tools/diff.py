from itertools import chain
from typing import Dict, Any
from pathlib import Path
import json
from termcolor import colored


class ObjectProxy:
    id: str
    module: str
    type: str
    identifier: str

    ignored: set[str]
    ignore: bool
    fields: Dict[str, Any]

    def __init__(self, data: Dict[str, Any]):
        self.id = data["id"]
        self.module = data["module"]
        self.type = data["type"]
        self.identifier = data["identifier"]
        self.ignore = data.get("ignore", False)
        self.ignored = set(data.get("ignored", []))
        self.fields = {}

    def __repr__(self):
        return f"{self.module}.{self.type}({self.identifier})"


Store = Dict[str, ObjectProxy]


def build_value(data, store: Store):
    if isinstance(data, dict):
        t = data.get("type", None)
        if t == "python":
            return store[data["value"]]
        elif t == "path":
            # Do not return anything
            return None  # Path(data["value"])
        elif t == "path.serialized":
            # Again, do not return anything (won't change the identifier)
            return None
        elif t is None:
            return {key: build_value(value, store) for key, value in data.items()}
        assert False, f"Data type {t} not handled"
    elif isinstance(data, list):
        return [build_value(value, store) for value in data]

    return data


def build_object(data: Dict[str, Any], store: Store) -> ObjectProxy:
    object = ObjectProxy(data)
    store[data["id"]] = object

    for key, value in data["fields"].items():
        object.fields[key] = build_value(value, store)

    return object


def load(path: Path):
    data = json.loads(path.read_text())
    store = {}
    for definition in data["objects"]:
        object = build_object(definition, store)

    return object


def print_diff(path: str, conf1: Any, conf2: Any):
    if type(conf1) != type(conf2):
        print(f"[{colored(path, 'red')}] {conf1} and {conf2} of different types")

    if isinstance(conf1, ObjectProxy) and isinstance(conf2, ObjectProxy):
        if conf1.identifier != conf2.identifier:
            print(
                f"[{colored(path, 'yellow')}] {conf1} and {conf2} have different identifiers"
            )
            keys = set(chain(conf1.fields.keys(), conf2.fields.keys()))
            for key in keys:
                print_diff(
                    f"{path}.{key}" if path else key,
                    conf1.fields.get(key, None),
                    conf2.fields.get(key, None),
                )

    elif isinstance(conf1, dict) and isinstance(conf2, dict):
        keys = set(chain(conf1.keys(), conf2.keys()))
        for key in keys:
            print_diff(
                f"{path}.{key}" if path else key,
                conf1.get(key, None),
                conf2.get(key, None),
            )

    elif isinstance(conf1, list) and isinstance(conf2, list):
        if len(conf1) != len(conf2):
            print(
                f"[{colored(path, 'red')}] {conf1} and {conf2} have not the same length"
            )
        else:
            for i in range(len(conf1)):
                print_diff(f"{path}.{i}" if path else key, conf1[i], conf2[i])

    elif conf1 != conf2:
        print(f"[{colored(path, 'red')}] {conf1} and {conf2} are not equal")


def diff(path1: Path, path2: Path):
    conf1, conf2 = load(path1), load(path2)

    print_diff("", conf1, conf2)
