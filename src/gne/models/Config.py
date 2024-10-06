from gne.__init__ import config


class Config:
    """
    Class to handle configuration of gNE.

    Attributes are grouped into a single dictionary of dictionaries.
    Values can be updated dynamically during instantiation or at runtime.

    Attributes:
    -----------
    config : dict
        A dictionary containing all the configuration settings
        grouped by their respective categories.
    """

    def __init__(self, **kwargs):
        """
        Parameters:
        -----------
        **kwargs : dict
            Key-value pairs where the key corresponds to a configuration category
            (e.g., 'source_geometry', 'optimizer'), and the value is a dictionary
            of settings to override the defaults specified in config.yaml
        """

        # Initialize the default configuration using the provided config object
        self.config = {
            "source_geometry": config["source_geometry"].copy(),
            "target_geometry": config["target_geometry"].copy(),
            "source_complex": config["source_complex"].copy(),
            "target_complex": config["target_complex"].copy(),
            "loss": config["loss"].copy(),
            "training": config["training"].copy(),
            "scheduler": config["scheduler"].copy(),
            "earlystop": config["earlystop"].copy(),
            "output": config["output"].copy(),
        }

        # Update the configuration with any values provided in kwargs
        for key, value in kwargs.items():
            updated = False
            # Check if the key directly matches one of the categories
            if key in self.config:
                self.config[key].update(value)
                updated = True
            else:
                # If no direct match, check within each sub-dictionary
                for sub_dict in self.config.values():
                    if key in sub_dict:
                        sub_dict[key] = value
                        updated = True
            if not updated:
                raise KeyError(f"Invalid config key: {key}")

    def __repr__(self):
        return (
            "gne.Config("
            + ", ".join(f"{key}={value}" for key, value in self.config.items())
            + ")"
        )

    def __getattr__(self, item):
        """
        Redefine the __getattr__ method to allow dynamic access to configurations
        using dot notation.
        (e.g., use obj.source_geometry instead of obj.config['source_geometry'])
        """
        if item in self.config:
            return self.config[item]
        raise AttributeError(f"'Config' object has no attribute '{item}'")
