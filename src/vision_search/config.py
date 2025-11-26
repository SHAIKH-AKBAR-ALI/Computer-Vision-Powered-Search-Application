import yaml

def load_config(config_path):
    """Loads a YAML config file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_config(config, config_path):
    """Saves a config dictionary to a YAML file."""
    with open(config_path, 'w') as f:
        yaml.dump(config, f)