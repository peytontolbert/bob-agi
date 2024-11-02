"""
This is the Discord environment, which is an application Bob can use.
"""
import logging
import time
import asyncio
import os
import docker
import subprocess
from app.env.browser import Browser
import discord
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC

class Discord:
    def __init__(self, browser: Browser):
        self.browser = browser
        self.client = None
        self.target_server_id = None
        self.connected = False
        self.discord_url = "https://discord.com/channels/@me"

    def launch(self):
        """Launch Discord web application"""
        try:
            # Navigate to Discord
            self.browser.navigate(self.discord_url)
            
            
            logging.info("Discord launched successfully")
            return True
            
        except Exception as e:
            logging.error(f"Failed to launch Discord: {e}")
            return False

    def login_with_token(self, token: str):
        """Login to Discord using token"""
        try:
            # Inject the token into local storage
            script = f"""
            localStorage.setItem('token', '"{token}"');
            """
            self.browser.webdriver.execute_script(script)
            
            # Reload the page to apply the token
            self.browser.webdriver.refresh()
            
            # Wait for login to complete
            self.browser.wait.until(
                lambda driver: driver.execute_script(
                    "return !!document.querySelector('[class*=\"privateChannels-\"]')"
                ),
                "Failed to login to Discord"
            )
            
            logging.info("Successfully logged into Discord")
            return True
            
        except Exception as e:
            logging.error(f"Failed to login to Discord: {e}")
            return False

    def go_to_voice_channel(self):
        pass

    def set_target_server(self, server_id):
        """
        Sets the target Discord server ID.
        """
        self.target_server_id = server_id

    def get_active_channels(self):
        """
        Returns a list of voice channels sorted by activity.
        """
        if not self.client or not self.target_server_id:
            return []
        
        server = self.client.get_guild(self.target_server_id)
        if not server:
            return []
        
        voice_channels = [channel for channel in server.channels 
                         if isinstance(channel, discord.VoiceChannel)]
        return voice_channels

    def join_channel(self, channel):
        """
        Joins the specified voice channel.
        """
        if not channel:
            return {'success': False, 'error': 'No channel specified'}
        
        try:
            # Use the browser to navigate to Discord
            self.browser.navigate_to(f'discord.com/channels/{self.target_server_id}/{channel.id}')
            
            # Click the join voice channel button
            join_button = self.browser.find_element('button[aria-label="Join Voice"]')
            if join_button:
                join_button.click()
                self.connected = True
                return {'success': True, 'channel': channel.name}
            
            return {'success': False, 'error': 'Join button not found'}
        except Exception as e:
            return {'success': False, 'error': str(e)}


