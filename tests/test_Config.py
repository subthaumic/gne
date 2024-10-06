import pytest
from gne.__init__ import config
from gne.models.Config import Config


def test_config_init():
    # Instantiate Config without any overrides
    cfg = Config()

    # Check that all the default config values are correctly loaded
    assert cfg.source_geometry == config["source_geometry"]
    assert cfg.target_geometry == config["target_geometry"]
    assert cfg.source_complex == config["source_complex"]
    assert cfg.target_complex == config["target_complex"]
    assert cfg.loss == config["loss"]
    assert cfg.training == config["training"]
    assert cfg.scheduler == config["scheduler"]
    assert cfg.earlystop == config["earlystop"]
    assert cfg.output == config["output"]

    # Instantiate Config without any overrides
    cfg = Config()


def test_config_init_with_dict_updates():
    # Instantiate Config with overrides
    cfg = Config(
        source_geometry={"dimension": 10},
        target_geometry={"dimension": 3, "initialization_method": "random"},
        loss={"loss_func": "L1"},
        training={"epochs": 200, "learning_rate": 0.1},
    )

    # Check that the overridden values are correctly set
    assert cfg.source_geometry["dimension"] == 10
    assert cfg.target_geometry["dimension"] == 3
    assert cfg.target_geometry["initialization_method"] == "random"
    assert cfg.loss["loss_func"] == "L1"
    assert cfg.training["epochs"] == 200
    assert cfg.training["learning_rate"] == 0.1

    # Check that other values remain as defaults
    assert cfg.source_geometry != config["source_geometry"]
    assert cfg.target_geometry != config["target_geometry"]
    assert cfg.source_complex == config["source_complex"]
    assert cfg.target_complex == config["target_complex"]
    assert cfg.loss != config["loss"]
    assert cfg.training != config["training"]
    assert cfg.scheduler == config["scheduler"]
    assert cfg.earlystop == config["earlystop"]
    assert cfg.output == config["output"]


def test_config_init_with_key_override():
    # Instantiate Config with a key override
    cfg = Config(k_neighbours=5, epochs=200)

    # Check that the overridden value is correctly set
    assert cfg.source_complex["k_neighbours"] == 5
    assert cfg.target_complex["k_neighbours"] == 5
    assert cfg.training["epochs"] == 200

    # Check that other values remain as defaults
    assert cfg.source_geometry == config["source_geometry"]
    assert cfg.target_geometry == config["target_geometry"]
    assert cfg.source_complex != config["source_complex"]
    assert cfg.target_complex != config["target_complex"]
    assert cfg.loss == config["loss"]
    assert cfg.training != config["training"]
    assert cfg.scheduler == config["scheduler"]
    assert cfg.earlystop == config["earlystop"]
    assert cfg.output == config["output"]


def test_config_init_with_invalid_key():
    # Test that providing an invalid key raises a KeyError
    with pytest.raises(KeyError):
        Config(invalid_key={"some_value": 123})


def test_dynamic_attribute_access():
    # Test dynamic attribute access using dot notation
    cfg = Config(
        target_geometry={"dimension": 3, "initialization_method": "random"},
        training={"epochs": 200, "learning_rate": 0.1},
    )

    assert cfg.target_geometry["initialization_method"] == "random"
    assert cfg.training["epochs"] == 200
    assert cfg.training["learning_rate"] == 0.1

    # Ensure that accessing an invalid attribute raises AttributeError
    with pytest.raises(AttributeError):
        _ = cfg.invalid_attribute
