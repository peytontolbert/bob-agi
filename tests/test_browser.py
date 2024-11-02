import pytest
from unittest.mock import Mock, patch
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from app.env.browser import Browser
from app.env.audio import Audio
from app.env.microphone import Microphone

@pytest.fixture
def mock_webdriver():
    with patch('selenium.webdriver.Edge') as mock_edge:
        driver = Mock()
        mock_edge.return_value = driver
        yield driver

@pytest.fixture
def browser():
    with patch('tkinter.Tk'):
        return Browser(audio=Mock(spec=Audio), microphone=Mock(spec=Microphone))

class TestBrowser:
    def test_init(self, browser):
        assert browser.webdriver is None
        assert browser.is_capturing is False
        assert browser._default_timeout == 10

    def test_launch_success(self, browser, mock_webdriver):
        # Arrange
        mock_webdriver.get_screenshot_as_png.return_value = b"fake_screenshot"
        
        # Act
        result = browser.launch()
        
        # Assert
        assert result is True
        assert browser.webdriver is not None
        assert browser.is_capturing is True

    def test_launch_failure(self, browser):
        with patch('selenium.webdriver.Edge', side_effect=Exception("WebDriver failed")):
            result = browser.launch()
            assert result is False
            assert browser.webdriver is None

    def test_find_element_success(self, browser, mock_webdriver):
        # Arrange
        browser.launch()
        mock_element = Mock()
        browser.wait.until.return_value = mock_element
        
        # Act
        element = browser.find_element(".test-selector")
        
        # Assert
        assert element == mock_element
        browser.wait.until.assert_called_once()

    def test_find_element_not_found(self, browser, mock_webdriver):
        # Arrange
        browser.launch()
        browser.wait.until.side_effect = TimeoutException()
        
        # Act
        element = browser.find_element(".test-selector")
        
        # Assert
        assert element is None

    def test_wait_until_loaded_success(self, browser, mock_webdriver):
        # Arrange
        browser.launch()
        browser.webdriver.execute_script.return_value = 'complete'
        
        # Act
        result = browser.wait_until_loaded()
        
        # Assert
        assert result is True

    def test_wait_until_loaded_timeout(self, browser, mock_webdriver):
        # Arrange
        browser.launch()
        browser.wait.until.side_effect = TimeoutException()
        
        # Act
        result = browser.wait_until_loaded()
        
        # Assert
        assert result is False

    def test_navigate(self, browser, mock_webdriver):
        # Arrange
        browser.launch()
        test_url = "https://example.com"
        
        # Act
        browser.navigate(test_url)
        
        # Assert
        browser.webdriver.get.assert_called_once_with(test_url)

    def test_close(self, browser, mock_webdriver):
        # Arrange
        browser.launch()
        
        # Act
        browser.close()
        
        # Assert
        assert browser.is_capturing is False
        browser.webdriver.quit.assert_called_once()
