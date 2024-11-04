import pytest
from pathlib import Path
from omegaconf import OmegaConf
import hydra

@pytest.fixture(scope="session")
def test_config():
    """Load test configuration"""
    config_dir = Path(__file__).parent / "conf"
    with hydra.initialize(config_path=str(config_dir.relative_to(Path.cwd()))):
        return hydra.compose(config_name="test_config")

@pytest.fixture(scope="session")
def screenshots_dir():
    """Provide screenshots directory path"""
    path = Path(__file__).parent / "screenshots"
    path.mkdir(exist_ok=True)
    return path

@pytest.fixture(scope="session")
def fixtures_dir():
    """Provide fixtures directory path"""
    path = Path(__file__).parent / "fixtures"
    path.mkdir(exist_ok=True)
    return path 