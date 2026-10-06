from pathlib import Path
import yaml


CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_config():

    with open(CONFIG_PATH, "r") as file:
        config = yaml.safe_load(file)

    return config