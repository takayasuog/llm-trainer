import dataclasses
import inspect
import os
import pathlib
import yaml


@dataclasses.dataclass
class Config:
    def print(self):
        print(
            f"\n\n\nLoRA fine-tuning params:\n")
        for key, val in dataclasses.asdict(self).items():
            print("-- ", key, ":", val)

    def save(self, config_path: str):
        """ Export config as YAML file """
        assert os.path.exists(config_path), f"directory {config_path.parent} does not exist"

        def _convert_dict(data):
            for key, val in data.items():
                if isinstance(val, pathlib.Path):
                    data[key] = str(val)
                elif isinstance(val, dict):
                    data[key] = _convert_dict(val)
            return data

        with open(config_path, 'w') as f:
            yaml.dump(_convert_dict(dataclasses.asdict(self)), f)

    @classmethod
    def load(cls, config_path: str):
        """ Load config from YAML file """
        assert os.path.exists(config_path), f"YAML config {config_path} does not exist"

        def _convert_from_dict(parent_cls, data):
            for key, val in data.items():
                child_class = parent_cls.__dataclass_fields__[key].type
                if child_class == pathlib.Path:
                    data[key] = pathlib.Path(val)
                if inspect.isclass(child_class) and issubclass(child_class, Config):
                    data[key] = child_class(**_convert_from_dict(child_class, val))
            return data

        with open(config_path) as f:
            config_data = yaml.full_load(f)
            # recursively convert config item to Config
            config_data = _convert_from_dict(cls, config_data)
            return cls(**config_data)