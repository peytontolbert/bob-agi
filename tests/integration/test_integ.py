from tests.test_base import BaseTest
from unittest.mock import patch
from app.env.computer import Computer
from app.bob.bob import Bob
import pytest

@pytest.mark.integration
class TestIntegration(BaseTest):
    def setUp(self):
        super().setUp()
        self.computer = Computer()
        self.bob = Bob(self.computer)

    @patch('hydra.main')
    def test_discord_integration(self, mock_hydra_main):
        """Test basic Discord integration functionality"""
        # Arrange
        mock_hydra_main.return_value = None
        discord_config = self.config.test.discord
        
        # Set up mock behaviors
        with patch.object(self.computer.discord, 'launch', return_value=True) as mock_launch, \
             patch.object(self.computer.discord, 'join_channel', return_value=self.config.mocks.discord.join_channel_response) as mock_join_channel:
            
            # Act
            launch_result = self.computer.discord.launch()
            join_result = self.computer.discord.join_channel({
                'id': discord_config.server_id,
                'name': discord_config.channel_name
            })
            
            # Assert
            mock_launch.assert_called_once()
            mock_join_channel.assert_called_once_with({
                'id': discord_config.server_id,
                'name': discord_config.channel_name
            })
            self.assertTrue(launch_result)
            self.assertTrue(join_result['success'])
            self.assertEqual(join_result['channel'], discord_config.channel_name)

    def tearDown(self):
        self.computer.shutdown()
        super().tearDown() 