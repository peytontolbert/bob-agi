import pytest
import sys
import os
import warnings

def pytest_configure(config):
    """Configure test environment before running tests"""
    # Add project root to Python path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, project_root)
    
    # Pre-import problematic modules in controlled way
    try:
        import numpy
    except ImportError:
        pass
        
    try:
        import torch
    except ImportError:
        pass
    
    # Suppress torch triton warnings
    warnings.filterwarnings(
        "ignore", 
        message="Only a single TORCH_LIBRARY can be used"
    )

@pytest.fixture(scope="session", autouse=True)
def cleanup_modules():
    """Cleanup modules after test session"""
    yield
    # Cleanup modules that might cause issues
    problematic_modules = ['torch', 'numpy', 'PIL']
    for module in problematic_modules:
        if module in sys.modules:
            del sys.modules[module] 