import unittest
from unittest.mock import MagicMock, patch
from app.env.computer.discord import Discord
from app.env.computer.browser import Browser
from selenium.common.exceptions import TimeoutException
import discord  # Add this import to access discord.VoiceChannel

class TestDiscord(unittest.TestCase):
    def setUp(self):
        self.mock_browser = MagicMock(spec=Browser)
        self.mock_browser.webdriver = MagicMock()
        self.discord = Discord(browser=self.mock_browser)
        self.discord.target_server_id = "123456789"  # Initialize target_server_id

    def test_init(self):
        self.assertEqual(self.discord.browser, self.mock_browser)
        self.assertIsNone(self.discord.root)
        self.assertIsNone(self.discord.screen)
        self.assertIsNone(self.discord.client)
        self.assertIsNone(self.discord.current_channel)
        self.assertIsNone(self.discord.current_server)
        self.assertFalse(self.discord.connected)
        self.assertEqual(self.discord.discord_url, "https://discord.com/channels/@me")

    def test_launch_success_login_screen(self):
        with patch.object(self.discord, 'browser') as mock_browser:
            # Setup the mock
            mock_browser.wait_until_loaded.return_value = True
            mock_browser.find_element.side_effect = [
                MagicMock(),  # Login element found
                TimeoutException()
            ]

            result = self.discord.launch()
            self.assertTrue(result)
            mock_browser.navigate.assert_called_with("https://discord.com/channels/@me")
            mock_browser.find_element.assert_called()

    def test_launch_success_main_interface(self):
        with patch.object(self.discord, 'browser') as mock_browser:
            # Setup the mock
            mock_browser.wait_until_loaded.return_value = True
            mock_browser.find_element.side_effect = [
                TimeoutException(),  # No login element
                MagicMock()          # Main interface element found
            ]

            result = self.discord.launch()
            self.assertTrue(result)
            mock_browser.navigate.assert_called_with("https://discord.com/channels/@me")
            mock_browser.find_element.assert_called()

    @patch('app.env.discord.logging')
    def test_launch_failure(self, mock_logging):
        # Setup the mock to raise an exception
        self.discord.browser.navigate.side_effect = Exception("Navigation failed")

        result = self.discord.launch()
        self.assertFalse(result)
        mock_logging.error.assert_called_with("Failed to launch Discord: Navigation failed")

    def test_login_with_token_success(self):
        with patch.object(self.discord, 'browser') as mock_browser:
            mock_browser.webdriver.execute_script = MagicMock()
            mock_browser.webdriver.refresh = MagicMock()
            mock_browser.wait.until = MagicMock(return_value=True)

            result = self.discord.login_with_token("fake_token")
            self.assertTrue(result)
            mock_browser.webdriver.execute_script.assert_called()
            mock_browser.webdriver.refresh.assert_called()
            mock_browser.wait.until.assert_called()

    @patch('app.env.discord.logging')
    def test_login_with_token_failure(self, mock_logging):
        # Setup the mock to raise an exception
        self.discord.browser.webdriver.execute_script.side_effect = Exception("Script injection failed")

        result = self.discord.login_with_token("fake_token")
        self.assertFalse(result)
        mock_logging.error.assert_called_with("Failed to login to Discord: Script injection failed")

    def test_set_target_server(self):
        self.discord.set_target_server("123456789")
        self.assertEqual(self.discord.target_server_id, "123456789")

    def test_get_active_channels_no_client(self):
        # Patch the 'client' attribute on the Discord instance
        with patch.object(self.discord, 'client', new=None):
            channels = self.discord.get_active_channels()
            self.assertEqual(channels, [])

    def test_get_active_channels_no_server(self):
        # Patch the 'client' attribute on the Discord instance
        with patch.object(self.discord, 'client') as mock_client:
            mock_client.get_guild.return_value = None
            channels = self.discord.get_active_channels()
            self.assertEqual(channels, [])

    def test_get_active_channels_success(self):
        # Patch the 'client' attribute on the Discord instance
        with patch.object(self.discord, 'client') as mock_client:
            mock_server = MagicMock()
            mock_channel = MagicMock(spec=discord.VoiceChannel)
            mock_server.channels = [mock_channel]
            mock_client.get_guild.return_value = mock_server

            channels = self.discord.get_active_channels()
            self.assertEqual(channels, [mock_channel])

    def test_join_channel_success(self):
        with patch.object(self.discord, 'browser') as mock_browser:
            mock_channel = MagicMock()
            mock_channel.id = "channel123"
            mock_channel.name = "General"

            mock_browser.navigate_to = MagicMock()
            mock_join_button = MagicMock()
            mock_browser.find_element.return_value = mock_join_button

            result = self.discord.join_channel(mock_channel)
            self.assertEqual(result, {'success': True, 'channel': "General"})
            mock_browser.navigate_to.assert_called_with('discord.com/channels/123456789/channel123')
            mock_join_button.click.assert_called()
            self.assertTrue(self.discord.connected)

    def test_join_channel_no_button(self):
        with patch.object(self.discord, 'browser') as mock_browser:
            mock_channel = MagicMock()
            mock_channel.id = "channel123"
            mock_channel.name = "General"

            mock_browser.navigate_to = MagicMock()
            mock_browser.find_element.return_value = None

            result = self.discord.join_channel(mock_channel)
            self.assertEqual(result, {'success': False, 'error': 'Join button not found'})
            mock_browser.navigate_to.assert_called_with('discord.com/channels/123456789/channel123')

    def test_join_channel_exception(self):
        with patch.object(self.discord, 'browser') as mock_browser:
            mock_channel = MagicMock()
            mock_channel.id = "channel123"

            mock_browser.navigate_to.side_effect = Exception("Navigation error")

            result = self.discord.join_channel(mock_channel)
            self.assertFalse(result['success'])
            self.assertIn("Navigation error", result['error'])

    @patch('app.env.discord.logging')
    def test_close(self, mock_logging):
        self.discord.root = MagicMock()
        self.discord.close()
        self.discord.root.destroy.assert_called()
        mock_logging.error.assert_not_called()

    @patch('app.env.discord.logging')
    def test_close_destroy_exception(self, mock_logging):
        self.discord.root = MagicMock()
        self.discord.root.destroy.side_effect = Exception("Destroy failed")
        self.discord.close()
        mock_logging.error.assert_called_with("Error destroying Discord root window: Destroy failed")

if __name__ == '__main__':
    unittest.main()
