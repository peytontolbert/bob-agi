import pytest
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import time

from app.bob.bob import Bob
from app.env.computer import Computer

@pytest.fixture
def computer():
    computer = Computer()
    computer.startup()  # Let Computer handle browser initialization
    yield computer
    computer.shutdown()  # Proper cleanup

@pytest.fixture
def bob(computer):
    return Bob(computer)

class TestBobEyesightBrowserPerception:
    
    def test_bob_can_perceive_basic_webpage(self, bob, computer):
        """Test that Bob can perceive and understand a basic webpage"""
        # Arrange
        computer.screen.browser.get("https://example.com")
        WebDriverWait(computer.screen.browser, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "h1"))
        )
        
        # Act
        env = bob._get_environment(computer)
        processed_env = bob.process_environment(env)
        
        # Assert
        assert processed_env['visual'] is not None
        assert processed_env['context']['scene_understanding'] is not None
        assert "Example Domain" in processed_env['context']['scene_understanding']
        assert any('heading' in str(elem['type']).lower() 
                  for elem in processed_env['visual'].get('elements', []))
        
    def test_bob_can_identify_interactive_elements(self, bob, computer):
        """Test that Bob can identify interactive elements like buttons and links"""
        # Arrange
        computer.screen.browser.get("https://www.wikipedia.org")
        WebDriverWait(computer.screen.browser, 10).until(
            EC.presence_of_element_located((By.ID, "searchInput"))
        )
        
        # Act
        env = bob._get_environment(computer)
        processed_env = bob.process_environment(env)
        
        # Assert
        elements = processed_env['visual'].get('elements', [])
        assert any(elem['type'] == 'search' for elem in elements), "Search box not found"
        assert any(elem['type'] == 'link' for elem in elements), "Links not found"
        assert any(elem['type'] == 'button' for elem in elements), "Buttons not found"
        
    def test_bob_understands_dynamic_content_changes(self, bob, computer):
        """Test that Bob can detect and understand dynamic content changes"""
        # Arrange
        browser = computer.screen.browser
        browser.get("https://www.wikipedia.org")
        search_box = WebDriverWait(browser, 10).until(
            EC.presence_of_element_located((By.ID, "searchInput"))
        )
        
        # Get initial perception
        initial_env = bob._get_environment(computer)
        initial_perception = bob.process_environment(initial_env)
        
        # Act - Trigger dynamic content
        search_box.send_keys("Python programming")
        try:
            WebDriverWait(browser, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "suggestion"))
            )
        except TimeoutException:
            pytest.skip("Search suggestions not available")
        
        # Get updated perception
        time.sleep(1)  # Brief pause to ensure visual processing
        new_env = bob._get_environment(computer)
        new_perception = bob.process_environment(new_env)
        
        # Assert
        assert initial_perception['context']['scene_understanding'] != \
               new_perception['context']['scene_understanding']
        assert "Python programming" in str(new_perception['context']['scene_understanding']).lower()
        assert any('suggestion' in str(elem.get('type', '')).lower() 
                  for elem in new_perception['visual'].get('elements', []))
        
    def test_bob_maintains_visual_memory(self, bob, computer):
        """Test that Bob maintains a memory of visual changes"""
        # Arrange
        browser = computer.screen.browser
        browser.get("https://example.com")
        WebDriverWait(browser, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "h1"))
        )
        
        # Act - Process multiple visual states
        visual_states = []
        for _ in range(3):
            env = bob._get_environment(computer)
            processed = bob.process_environment(env)
            visual_states.append(processed['visual'])
            time.sleep(0.5)  # Brief pause between captures
            
        # Assert
        visual_memory = bob.interaction_context['visual_memory']
        assert len(visual_memory) >= 3, "Visual memory not maintained"
        assert all('scene_understanding' in state for state in visual_memory)
        assert all('timestamp' in state for state in visual_memory)
        
    def test_bob_integrates_visual_and_cognitive_context(self, bob, computer):
        """Test that Bob integrates visual perception with cognitive context"""
        # Arrange
        browser = computer.screen.browser
        browser.get("https://www.wikipedia.org")
        WebDriverWait(browser, 10).until(
            EC.presence_of_element_located((By.ID, "searchInput"))
        )
        
        # Set cognitive context
        bob.interaction_context['current_goal'] = "Find the search box on Wikipedia"
        
        # Act
        env = bob._get_environment(computer)
        processed_env = bob.process_environment(env)
        action = bob.decide_action(processed_env)
        
        # Assert
        assert action is not None
        assert action['type'] == 'interaction'
        assert 'search' in str(action.get('element', {})).lower()
        assert processed_env['context'].get('integrated_understanding') is not None
        assert 'search' in str(processed_env['context']['integrated_understanding']).lower()
        
    def test_bob_handles_page_load_errors(self, bob, computer):
        """Test that Bob can handle and understand page load errors"""
        # Arrange
        browser = computer.screen.browser
        browser.get("https://thiswebsitedoesnotexist.com/")
        time.sleep(2)  # Wait for error page
        
        # Act
        env = bob._get_environment(computer)
        processed_env = bob.process_environment(env)
        
        # Assert
        assert processed_env['visual'] is not None
        assert 'error' in str(processed_env['context']['scene_understanding']).lower() or \
               'unable to connect' in str(processed_env['context']['scene_understanding']).lower()
        
    @pytest.mark.parametrize("viewport_size", [
        (800, 600),
        (1280, 720),
        (1920, 1080)
    ])
    def test_bob_adapts_to_different_viewport_sizes(self, bob, computer, viewport_size):
        """Test that Bob can adapt to different viewport sizes"""
        # Arrange
        width, height = viewport_size
        browser = computer.screen.browser
        browser.set_window_size(width, height)
        browser.get("https://example.com")
        
        # Act
        env = bob._get_environment(computer)
        processed_env = bob.process_environment(env)
        
        # Assert
        assert processed_env['visual'] is not None
        assert processed_env['visual'].get('viewport_size') == {'width': width, 'height': height}
        assert processed_env['context']['scene_understanding'] is not None
