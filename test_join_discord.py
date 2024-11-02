### This script is for testing the integration of Bob and his computer environment. This script will hard-code Bob's task of joining Agora's Discord voice channel. ###

import logging
from app.env.discord import Discord
import discord
from app.env.computer import Computer
from app.bob.bob import Bob

def main():
    print("Starting Bob...")
    computer = Computer()
    print("Computer initialized.")
    
    # Configure Discord settings
    AGORA_SERVER_ID = "999382051935506503"  # Replace with actual Agora server ID
    
    # Initialize Bob with the specific task
    bob = Bob(computer)
    
    # Configure Discord client
    computer.discord.set_target_server(AGORA_SERVER_ID)
    
    # Create a task for Bob to join Discord
    task = {
        'type': 'join_voice',
        'server': AGORA_SERVER_ID,
        'action': 'join_active_channel'  # This will make Bob join the channel with most people
    }
    
    # Add the task to Bob's thoughts
    bob.thoughts._add_to_thought_queue(task)
    
    # Start Bob's main loop
    try:
        bob.alive(computer)
    except KeyboardInterrupt:
        print("Shutting down Bob...")
        computer.shutdown()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()