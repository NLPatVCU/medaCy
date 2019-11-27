import json
import os

_this_dir = os.path.dirname(__file__)
_config_path = os.path.join(_this_dir, "../../config.json")


def _validate_path(path):
    """Raises an error if the provided path is not valid"""
    if not os.path.isfile(path):
        raise FileNotFoundError("The MetaMap path provided is not a file")
    if not path.endswith("metamap"):
        raise ValueError("The name of this file is not 'metamap', therefore it cannot be the MetaMap binary")


def get_metamap():
    """
    Gets the path to MetaMap from the config json and returns it, or prompts
    the user to specify the MetaMap path if it has not been set for this installation
    """
    with open(_config_path, 'rb') as f:
        config_data = json.load(f)

    mm_path = config_data['metamap_path']

    # 0 is the default value, indicating that the path has never been specified
    if mm_path == 0:
        print("The path to MetaMap has not been specified for this installation")
        new_path = input("Please specify path to the MetaMap binary: ")

        _validate_path(new_path)

        config_data['metamap_path'] = new_path

        with open(_config_path, 'w') as f:
            json.dump(config_data, f)

        mm_path = new_path

    elif not os.path.isfile(mm_path):
        new_path = input("MetaMap was not found at the given location; please specify a new path: ")
        _validate_path(new_path)
        mm_path = new_path

    return mm_path


def get_metamap_path():
    """Returns the path to the MetaMap binary, or 0 if it has not been set."""
    with open(_config_path, 'rb') as f:
        config_data = json.load(f)

    return config_data['metamap_path']