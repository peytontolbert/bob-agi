import pytest
from unittest.mock import Mock, patch
from app.env.computer.computer import Computer
from app.env.computer.screen import Screen
from app.env.computer.audio import Audio
from app.env.computer.microphone import Microphone
from app.env.computer.keyboard import Keyboard
from app.env.computer.mouse import Mouse
from app.env.computer.browser import Browser
from app.env.computer.discord import Discord

@pytest.fixture
def mock_components():
    """Fixture to create mock components"""
    return {
        'screen': Mock(spec=Screen),
        'audio': Mock(spec=Audio),
        'microphone': Mock(spec=Microphone),
        'keyboard': Mock(spec=Keyboard),
        'mouse': Mock(spec=Mouse),
        'browser': Mock(spec=Browser),
        'discord': Mock(spec=Discord)
    }

@pytest.fixture
def computer(mock_components):
    """Fixture to create a Computer instance with mocked components"""
    with patch('app.env.computer.Screen', return_value=mock_components['screen']), \
         patch('app.env.computer.Audio', return_value=mock_components['audio']), \
         patch('app.env.computer.Microphone', return_value=mock_components['microphone']), \
         patch('app.env.computer.Keyboard', return_value=mock_components['keyboard']), \
         patch('app.env.computer.Mouse', return_value=mock_components['mouse']), \
         patch('app.env.computer.Browser', return_value=mock_components['browser']), \
         patch('app.env.computer.Discord', return_value=mock_components['discord']):
        computer = Computer()
        # Set the mocked browser webdriver to simulate successful initialization
        mock_components['browser'].webdriver = True
        return computer, mock_components

def test_computer_initialization(computer):
    """Test if Computer initializes all components correctly"""
    computer_instance, mocks = computer
    
    assert computer_instance.screen == mocks['screen']
    assert computer_instance.audio == mocks['audio']
    assert computer_instance.microphone == mocks['microphone']
    assert computer_instance.keyboard == mocks['keyboard']
    assert computer_instance.mouse == mocks['mouse']
    assert computer_instance.apps['browser'] == mocks['browser']
    assert computer_instance.discord is None  # Discord starts as None

def test_computer_startup_success(computer):
    """Test successful computer startup sequence"""
    computer_instance, mocks = computer
    
    # Configure mocks
    mocks['browser'].launch.return_value = True
    mocks['browser'].webdriver = True  # Simulate browser webdriver initialization
    
    # Configure Discord mock to return itself from launch
    discord_mock = mocks['discord']
    discord_mock.launch.return_value = True
    
    # We need to patch Discord initialization since it's created in startup()
    with patch('app.env.computer.Discord', return_value=discord_mock) as mock_discord_class:
        # Run startup
        computer_instance.startup()
        
        # Verify component initialization
        mocks['screen'].initialize.assert_called_once()
        mocks['audio'].initialize.assert_called_once()
        mocks['microphone'].initialize.assert_called_once()
        
        # Verify app launches
        mocks['browser'].launch.assert_called_once()
        
        # Verify Discord was initialized with the browser
        mock_discord_class.assert_called_once_with(browser=mocks['browser'])
        
        # Verify Discord instance was properly stored and launched
        assert computer_instance.discord == discord_mock
        assert computer_instance.apps['discord'] == discord_mock
        assert discord_mock.launch.call_count == 1

def test_computer_startup_browser_failure(computer):
    """Test computer startup handling browser launch failure"""
    computer_instance, mocks = computer
    
    # Configure browser launch to fail
    mocks['browser'].launch.return_value = False
    
    # Verify startup raises exception
    with pytest.raises(Exception, match="Failed to launch browser"):
        computer_instance.startup()

def test_launch_app_success(computer):
    """Test successful app launch"""
    computer_instance, mocks = computer
    
    mocks['browser'].launch.return_value = True
    assert computer_instance.launch_app('browser') is True
    mocks['browser'].launch.assert_called_once()

def test_launch_app_failure(computer):
    """Test failed app launch"""
    computer_instance, mocks = computer
    
    # Configure app launch to fail
    mocks['browser'].launch.return_value = False
    assert computer_instance.launch_app('browser') is False
    mocks['browser'].launch.assert_called_once()

def test_launch_nonexistent_app(computer):
    """Test launching non-existent app"""
    computer_instance, _ = computer
    assert computer_instance.launch_app('nonexistent') is False

def test_close_app(computer):
    """Test closing an app"""
    computer_instance, mocks = computer
    
    computer_instance.close_app('browser')
    mocks['browser'].close.assert_called_once()

def test_shutdown(computer):
    """Test computer shutdown sequence"""
    computer_instance, mocks = computer
    
    computer_instance.shutdown()
    
    # Verify input devices are stopped
    mocks['keyboard'].stop.assert_called_once()
    mocks['mouse'].stop.assert_called_once()

def test_get_system_state(computer):
    """Test getting system state"""
    computer_instance, _ = computer
    
    # Mock the required methods
    computer_instance.get_resource_usage = lambda: {
        'cpu': 0.0,
        'memory': 0.0,
        'disk': 0.0
    }
    computer_instance.get_pending_notifications = lambda: []
    computer_instance.get_interaction_mode = lambda: 'normal'
    
    state = computer_instance.get_system_state()
    
    assert isinstance(state, dict)
    assert 'running_apps' in state
    assert 'resource_usage' in state
    assert 'notifications' in state
    assert 'interaction_mode' in state
    assert 'browser' in state['running_apps']
    
    # Verify the structure of resource usage
    assert 'cpu' in state['resource_usage']
    assert 'memory' in state['resource_usage']
    assert 'disk' in state['resource_usage']
