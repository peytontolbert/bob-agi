import pytest
from unittest.mock import Mock, patch
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException, WebDriverException
from app.env.browser import Browser
from app.env.audio import Audio
from app.env.microphone import Microphone
from selenium.webdriver.support.ui import WebDriverWait

@pytest.fixture
def mock_webdriver():
    with patch('app.env.browser.WebDriverWait') as mock_wait:
        driver = Mock()
        with patch('selenium.webdriver.Edge') as mock_edge:
            mock_edge.return_value = driver
            driver.get_screenshot_as_png.return_value = (
                b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x10'
                b'\x00\x00\x00\x10\x08\x02\x00\x00\x00\x90wS\xde\x00'
                b'\x00\x00\x0cIDATx\x9cc`\x00\x00\x00\x02\x00\x01\xe2!'
                b'\xbc\x33\x00\x00\x00\x00IEND\xaeB`\x82'
            )
            wait_instance = Mock(spec=WebDriverWait)
            mock_wait.return_value = wait_instance
            driver.wait = wait_instance
            yield driver

@pytest.fixture
def browser(mock_webdriver):
    with patch('tkinter.Tk'):
        return Browser(audio=Mock(spec=Audio), microphone=Mock(spec=Microphone))

class TestBrowser:
    def test_init(self, browser):
        assert browser.webdriver is None
        assert browser.is_capturing is False
        assert browser._default_timeout == 10

    def test_launch_success(self, browser, mock_webdriver):
        # Arrange
        mock_webdriver.get_screenshot_as_png.return_value = (
            b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01'
            b'\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89'
            b'\x00\x00\x00\nIDATx\xdac\xf8\x0f\x00\x01\x01\x01\x00'
            b'\x18\xdd\x8d\x7d\x00\x00\x00\x00IEND\xaeB`\x82'
        )
        # Ensure that the screenshot processing does not fail
        mock_webdriver.get_screenshot_as_png.return_value = (
            b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x10'
            b'\x00\x00\x00\x10\x08\x02\x00\x00\x00\x90wS\xde\x00'
            b'\x00\x00\x0cIDATx\x9cc`\x00\x00\x00\x02\x00\x01\xe2!'
            b'\xbc\x33\x00\x00\x00\x00IEND\xaeB`\x82'
        )

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
        browser.webdriver.wait.until.return_value = mock_element
        
        # Act
        element = browser.find_element(".test-selector")
        
        # Assert
        assert element == mock_element
        browser.webdriver.wait.until.assert_called_once()

    def test_find_element_not_found(self, browser, mock_webdriver):
        # Arrange
        browser.launch()
        browser.webdriver.wait.until.side_effect = TimeoutException()
        
        # Act
        element = browser.find_element(".test-selector")
        
        # Assert
        assert element is None

    def test_wait_until_loaded_timeout(self, browser, mock_webdriver):
        # Arrange
        browser.launch()
        browser.webdriver.wait.until.side_effect = TimeoutException()
        
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

    def test_move_mouse(self, browser, mock_webdriver):
        # Arrange
        browser.launch()
        x, y = 150, 250
        speed = 2.0

        # Act
        browser.move_mouse(x, y, speed)

        # Assert
        # Ensure execute_script is called on the webdriver
        mock_webdriver.execute_script.assert_called_once_with(
            f"window.moveMouse({x}, {y}, {speed})"
        )

    def test_click_mouse_left(self, browser, mock_webdriver):
        # Arrange
        browser.launch()
        button = 'left'

        # Act
        browser.click_mouse(button)

        # Assert
        # Ensure execute_script is called on the webdriver
        mock_webdriver.execute_script.assert_called_once_with(
            f"window.clickMouse('{button}')"
        )

    def test_click_mouse_invalid_button(self, browser, mock_webdriver):
        # Arrange
        browser.launch()
        invalid_button = 'invalid'

        # Act & Assert
        with pytest.raises(ValueError):
            browser.click_mouse(invalid_button)

    def test_set_screen(self, browser):
        # Arrange
        mock_screen = Mock()

        # Act
        browser.set_screen(mock_screen)

        # Assert
        assert browser.screen == mock_screen

    def test_navigate_exception(self, browser, mock_webdriver):
        # Arrange
        browser.launch()
        test_url = "https://example.com"
        mock_webdriver.get.side_effect = WebDriverException("Navigation failed")

        # Act
        with pytest.raises(WebDriverException):
            browser.navigate(test_url)

        # Assert
        mock_webdriver.get.assert_called_once_with(test_url)

    def test_wait_until_loaded_success(self, browser, mock_webdriver):
        # Arrange
        browser.launch()
        mock_webdriver.execute_script.return_value = 'complete'
        # Configure `wait.until` to execute the lambda, simulating successful page load
        mock_webdriver.wait.until.side_effect = lambda func: func(mock_webdriver)

        # Act
        result = browser.wait_until_loaded()

        # Assert
        assert result is True
        mock_webdriver.execute_script.assert_called_once_with('return document.readyState')

    def test_wait_until_loaded_failure(self, browser, mock_webdriver):
        # Arrange
        browser.launch()
        mock_webdriver.execute_script.return_value = 'loading'
        # Configure `wait.until` to raise a TimeoutException, simulating a timeout
        mock_webdriver.wait.until.side_effect = TimeoutException()

        # Act
        result = browser.wait_until_loaded(timeout=1)

        # Assert
        assert result is False
