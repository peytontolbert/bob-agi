"""Integration test for Bob joining Discord voice channel"""

import pytest
import logging
from app.env.discord import Discord
from app.env.computer import Computer
from app.bob.bob import Bob
import asyncio
import time

@pytest.fixture
def setup_computer():
    """Fixture to set up computer environment"""
    computer = Computer()
    computer.run()
    return computer

@pytest.fixture
def setup_bob(setup_computer):
    """Fixture to set up Bob with computer"""
    bob = Bob(setup_computer)
    return bob

@pytest.fixture
def setup_discord(setup_computer):
    """Fixture to set up Discord environment"""
    discord = setup_computer.discord
    # Configure Discord settings
    AGORA_SERVER_ID = "999382051935506503"  # Replace with actual Agora server ID
    discord.set_target_server(AGORA_SERVER_ID)
    return discord

@pytest.mark.asyncio
async def test_join_discord_voice_channel(setup_computer, setup_bob, setup_discord):
    """Test Bob's ability to join Discord voice channel"""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Set Bob's goal to join Discord voice channel
        setup_bob.interaction_context['current_goal'] = {
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
        
        
        # Start Bob's processing
        start_time = time.time()
        timeout = 30  # 30 second timeout
        
        while time.time() - start_time < timeout:
            # Get environmental input
            env = setup_bob._get_environment(setup_computer)
            
            # Process environment
            processed_env = setup_bob.process_environment(env)
            
            # Let Bob decide and execute action
            action = setup_bob.decide_action(processed_env)
            if action:
                result = setup_bob.execute_action(action)
                
                # Check if we successfully joined voice channel
                if result and result.get('success'):
                    if setup_discord.connected:
                        logger.info("Successfully joined voice channel")
                        break
            
            await asyncio.sleep(0.1)
            
        # Verify we joined the voice channel
        assert setup_discord.connected, "Failed to join voice channel"
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise
        
    finally:
        # Cleanup
        setup_computer.shutdown()

@pytest.mark.asyncio
async def test_discord_error_handling(setup_computer, setup_bob, setup_discord):
    """Test error handling when joining Discord fails"""
    
    try:
        # Set invalid server ID to force error
        setup_discord.set_target_server("invalid_server_id")
        
        # Set Bob's goal
        setup_bob.interaction_context['current_goal'] = {
            'type': 'join_voice_channel',
            'target': 'Invalid Server',
            'application': 'Discord'
        }
        
        # Try to join invalid server
        await asyncio.sleep(2)
        
        # Process for a short time
        start_time = time.time()
        timeout = 10
        
        while time.time() - start_time < timeout:
            env = setup_bob._get_environment(setup_computer)
            processed_env = setup_bob.process_environment(env)
            action = setup_bob.decide_action(processed_env)
            if action:
                result = setup_bob.execute_action(action)
                assert not result.get('success'), "Should not succeed with invalid server"
            await asyncio.sleep(0.1)
            
        assert not setup_discord.connected, "Should not connect to invalid server"
        
    finally:
        setup_computer.shutdown()

def test_discord_visual_verification(setup_computer, setup_bob, setup_discord):
    """Test Bob's visual verification of Discord interface"""
    
    try:
        # Get screen state through Bob's eyes
        screen_state = setup_bob.eyes.capture_screen()
        
        # Verify Bob can see Discord interface
        assert screen_state is not None
        assert 'frame' in screen_state
        
        # Use vision agent to verify Discord interface
        scene_understanding = setup_bob.vision_agent.perceive_scene(screen_state['frame'])
        
        assert scene_understanding is not None
        assert scene_understanding['status'] == 'success'
        assert 'Discord' in scene_understanding.get('description', '').lower()
        
    finally:
        setup_computer.shutdown()