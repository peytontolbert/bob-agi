"""
Script to initialize Bob's environment and have him join the Agora discord voice channel.
"""
from app.env.computer.computer import Computer
from app.bob.bob import Bob
import logging

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize computer and Bob
        logger.info("Initializing computer environment...")
        computer = Computer()
        
        logger.info("Starting Bob...")
        bob = Bob(computer)
        
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
        logger.info("Starting main processing loop...")
        bob.alive(computer)
        
    except Exception as e:
        logger.error(f"Error in main: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
