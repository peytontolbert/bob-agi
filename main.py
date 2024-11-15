"""
Script to initialize Bob's environment and have him join the Agora discord voice channel.
"""
from app.env.computer.computer import Computer
from pipelines.join_agora_voice_channel import join_agora_voice_channel
from app.bob.test_bob import Bob
import logging

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        # Initialize computer and Bob
        logger.info("Initializing computer environment...")
        computer = Computer()
        computer.startup()
        logger.info("Joining Agora voice channel...")
        join_agora_voice_channel(computer.browser)
        logger.info("Starting Bob...")

        bob = Bob(computer)
        
        
        logger.info("Goal set: Join Agora Discord voice channel")
        
        # Start Bob's main loop
        logger.info("Starting main processing loop...")
        bob.alive(computer)
        
    except Exception as e:
        logger.error(f"Error in main: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
