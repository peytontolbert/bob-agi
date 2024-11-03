### This script is for testing the integration of Bob and his computer environment. This script will hard-code Bob's task of joining Agora's Discord voice channel. ###

import logging
from app.env.discord import Discord
import discord
from app.env.computer import Computer
from app.bob.bob import Bob



def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    print("Starting Bob...")
    computer = Computer()
    print("Computer initialized.")
    
    # Configure Discord settings
    AGORA_SERVER_ID = "999382051935506503"  # Replace with actual Agora server ID
    
    # Initialize Bob with the specific task
    bob = Bob(computer)
    
    # Configure Discord client
    computer.discord.set_target_server(AGORA_SERVER_ID)
    

    # Set Bob's current goal to join Discord voice channel
    bob.interaction_context['current_goal'] = {
        'type': 'join_voice_channel',
        'target': 'Agora',
        'application': 'Discord',
        'steps': [
            'open_discord',
            'navigate_to_agora_server',
            'find_voice_channel',
            'join_voice_channel'
        ]
    }
    
    logger.info("Goal set: Join Agora Discord voice channel")
    
    # Start Bob's main loop
    try:
        bob.alive(computer)
    except KeyboardInterrupt:
        print("Shutting down Bob...")
        computer.shutdown()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()