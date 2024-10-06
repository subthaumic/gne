import yaml
from pathlib import Path

config_path = Path(__file__).resolve().parent / "config.yaml"

with config_path.open() as config_file:
    config = yaml.safe_load(config_file)
