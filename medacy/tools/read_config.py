import json
import os

from medacy import __file__ as medacy_root


_config_path = os.path.join(os.path.dirname(medacy_root), "../config.json")
assert os.path.isfile(_config_path)


def read_config(key):
    """
    Reads medaCy's config file and returns the value at a given key
    :param key: the desired key
    :return: the value at that key
    """
    with open(_config_path, "rb") as f:
        config = json.load(f)

    return config[key]

