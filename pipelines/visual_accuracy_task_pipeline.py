"""
Pipeline for accurately locating UI elements and completing the task of joining Discord voice channel.
"""
import logging
import time
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
from datetime import datetime

from pipelines.visual_accuracy_pipeline import VisualAccuracyPipeline, DetectionResult

@dataclass
class TaskStep:
    """Represents a step in the task pipeline"""
    description: str
    element_identifiers: List[str]  # Multiple ways to identify the element
    visual_cues: List[str]  # Visual characteristics to look for
    action: str  # click, type, wait
    expected_next: List[str]  # Multiple possible next states
    timeout: float = 10.0
    retry_count: int = 3
    confidence_threshold: float = 0.7

class DiscordVoiceChannelPipeline:
    def __init__(self, vision_agent, eyes, hands):
        self.vision_pipeline = VisualAccuracyPipeline(vision_agent, eyes)
        self.hands = hands
        self.eyes = eyes
        
        # Configure task steps with more detailed element identification
        self.task_steps = [
            TaskStep(
                description="Find Continue in Browser button",
                element_identifiers=[
                    "Continue in browser",
                    "Open Discord in your browser",
                    "Use Discord in browser"
                ],
                visual_cues=[
                    "blue button",
                    "clickable link",
                    "text with browser icon"
                ],
                action="click",
                expected_next=[
                    "Discord login page",
                    "Login with Discord",
                    "Welcome back to Discord"
                ],
                timeout=5.0,
                confidence_threshold=0.65
            ),
            TaskStep(
                description="Find and click login button",
                element_identifiers=[
                    "Login",
                    "Login with Discord",
                    "Log In",
                    "Sign In"
                ],
                visual_cues=[
                    "blue login button",
                    "prominent button at center",
                    "login form submit button"
                ],
                action="click",
                expected_next=[
                    "Discord main interface",
                    "Server list",
                    "Discord channels"
                ],
                timeout=8.0,
                confidence_threshold=0.75
            ),
            TaskStep(
                description="Locate Agora server",
                element_identifiers=[
                    "Agora server",
                    "Agora",
                    "Agora Discord server"
                ],
                visual_cues=[
                    "server icon on left sidebar",
                    "server name or logo",
                    "Agora community server"
                ],
                action="click",
                expected_next=[
                    "Agora server channels",
                    "Agora server content",
                    "Voice channels list"
                ],
                timeout=5.0
            ),
            TaskStep(
                description="Find Voice Channels section",
                element_identifiers=[
                    "Voice Channels",
                    "VOICE CHANNELS",
                    "Voice"
                ],
                visual_cues=[
                    "channel category header",
                    "expandable section",
                    "voice channel list"
                ],
                action="wait",
                expected_next=[
                    "Voice channel list",
                    "Available voice channels",
                    "Voice chat options"
                ],
                timeout=3.0
            ),
            TaskStep(
                description="Join Agora voice channel",
                element_identifiers=[
                    "Agora Voice",
                    "General Voice",
                    "Voice Chat"
                ],
                visual_cues=[
                    "speaker icon",
                    "voice channel entry",
                    "joinable voice channel"
                ],
                action="click",
                expected_next=[
                    "Connected to voice",
                    "Voice connected",
                    "In voice channel"
                ],
                timeout=5.0
            )
        ]
        
        self.results = []
        self.current_step = 0

    def _find_element_with_pipeline(self, step: TaskStep) -> Optional[DetectionResult]:
        """
        Uses visual accuracy pipeline to find element with multiple attempts and strategies.
        """
        best_result = None
        highest_confidence = 0
        
        # Try each identifier with visual cues
        for identifier in step.element_identifiers:
            for visual_cue in step.visual_cues:
                try:
                    # Combine identifier and visual cue for better targeting
                    search_query = f"{identifier} - {visual_cue}"
                    
                    # Use visual accuracy pipeline to find element
                    result = self.vision_pipeline.find_element_with_retry(
                        search_query,
                        confidence_threshold=step.confidence_threshold,
                        max_retries=2
                    )
                    
                    if result and result.confidence > highest_confidence:
                        highest_confidence = result.confidence
                        best_result = result
                        
                        # If confidence is very high, return immediately
                        if highest_confidence > 0.9:
                            return best_result
                            
                except Exception as e:
                    logging.warning(f"Error finding element with {search_query}: {e}")
                    continue
                    
        return best_result if highest_confidence >= step.confidence_threshold else None

    def _execute_step(self, step: TaskStep) -> Dict[str, Any]:
        """
        Executes a single step with enhanced element detection and verification.
        """
        start_time = time.time()
        
        for attempt in range(step.retry_count):
            try:
                # Find element using visual accuracy pipeline
                detection_result = self._find_element_with_pipeline(step)
                
                if not detection_result:
                    if attempt < step.retry_count - 1:
                        logging.warning(f"Element not found, attempt {attempt + 1}/{step.retry_count}")
                        time.sleep(1.0)
                        continue
                    return {
                        'success': False,
                        'step': step.description,
                        'error_message': f"Element not found after {step.retry_count} attempts",
                        'time_taken': time.time() - start_time
                    }
                
                # Execute action based on type
                if step.action == "click":
                    click_success = self.hands.click_element(detection_result.coordinates)
                    
                    if not click_success:
                        if attempt < step.retry_count - 1:
                            logging.warning(f"Click failed, attempt {attempt + 1}/{step.retry_count}")
                            time.sleep(1.0)
                            continue
                        return {
                            'success': False,
                            'step': step.description,
                            'error_message': "Click action failed",
                            'time_taken': time.time() - start_time
                        }
                
                # Verify next state with multiple possible states
                if not self._verify_next_state(step.expected_next, step.timeout):
                    if attempt < step.retry_count - 1:
                        logging.warning(f"Next state verification failed, attempt {attempt + 1}/{step.retry_count}")
                        time.sleep(1.0)
                        continue
                    return {
                        'success': False,
                        'step': step.description,
                        'error_message': "Failed to verify next state",
                        'time_taken': time.time() - start_time
                    }
                
                return {
                    'success': True,
                    'step': step.description,
                    'time_taken': time.time() - start_time,
                    'detection_confidence': detection_result.confidence,
                    'element_location': detection_result.coordinates
                }
                
            except Exception as e:
                if attempt < step.retry_count - 1:
                    logging.warning(f"Step execution failed, attempt {attempt + 1}/{step.retry_count}: {e}")
                    time.sleep(1.0)
                    continue
                return {
                    'success': False,
                    'step': step.description,
                    'error_message': str(e),
                    'time_taken': time.time() - start_time
                }

    def _verify_next_state(self, expected_states: List[str], timeout: float) -> bool:
        """
        Verifies any of the expected next states is reached.
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Try to verify each possible next state
            for expected_state in expected_states:
                try:
                    # Use visual accuracy pipeline to verify state
                    verification_result = self.vision_pipeline.verify_screen_state(
                        expected_state,
                        confidence_threshold=0.6  # Lower threshold for state verification
                    )
                    
                    if verification_result.success:
                        return True
                        
                except Exception as e:
                    logging.debug(f"Error verifying state {expected_state}: {e}")
                    continue
                    
            time.sleep(0.5)
            
        return False

    def execute_pipeline(self) -> Dict[str, Any]:
        """
        Executes the full pipeline to join Discord voice channel.
        
        Returns:
            Dict containing execution results and metrics
        """
        start_time = time.time()
        pipeline_results = {
            'success': False,
            'steps_completed': 0,
            'total_steps': len(self.task_steps),
            'step_results': [],
            'total_time': 0,
            'errors': []
        }

        try:
            for i, step in enumerate(self.task_steps):
                self.current_step = i
                logging.info(f"Executing step {i+1}/{len(self.task_steps)}: {step.description}")
                
                step_result = self._execute_step(step)
                pipeline_results['step_results'].append(step_result)
                
                if not step_result['success']:
                    error_msg = f"Failed at step {i+1}: {step_result.get('error_message', 'Unknown error')}"
                    pipeline_results['errors'].append(error_msg)
                    logging.error(error_msg)
                    break
                    
                pipeline_results['steps_completed'] += 1
                
                # Add delay between steps
                time.sleep(1.0)
                
            pipeline_results['success'] = (
                pipeline_results['steps_completed'] == len(self.task_steps)
            )
            
        except Exception as e:
            error_msg = f"Pipeline execution failed: {str(e)}"
            pipeline_results['errors'].append(error_msg)
            logging.error(error_msg)
            
        finally:
            pipeline_results['total_time'] = time.time() - start_time
            self._save_pipeline_results(pipeline_results)
            
        return pipeline_results

    def _save_pipeline_results(self, results: Dict[str, Any]):
        """Saves pipeline execution results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Add to vision pipeline results directory
        self.vision_pipeline._save_test_results({
            'pipeline_type': 'discord_voice_channel',
            'timestamp': timestamp,
            'results': results
        })

    def get_progress(self) -> Dict[str, Any]:
        """
        Returns current pipeline progress.
        
        Returns:
            Dict containing progress metrics
        """
        return {
            'current_step': self.current_step + 1,
            'total_steps': len(self.task_steps),
            'completed_steps': [r for r in self.results if r['success']],
            'success_rate': len([r for r in self.results if r['success']]) / max(1, len(self.results))
        }

if __name__ == "__main__":
    from app.agents.vision import VisionAgent
    from app.env.screen import Screen
    from app.env.computer import Computer
    from app.actions.eyesight import Eyesight
    from app.actions.hands import Hands
    import logging
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Initialize components
        computer = Computer()
        computer.run()
        eyes = Eyesight(computer.screen)
        hands = Hands(computer.mouse, computer.keyboard)
        vision_agent = VisionAgent()
        
        # Create pipeline
        pipeline = DiscordVoiceChannelPipeline(vision_agent, eyes, hands)
        
        # Execute pipeline
        logging.info("Starting Discord voice channel pipeline...")
        results = pipeline.execute_pipeline()
        
        # Log results
        logging.info("\nPipeline Results:")
        logging.info(f"Success: {results['success']}")
        logging.info(f"Steps completed: {results['steps_completed']}/{results['total_steps']}")
        logging.info(f"Total time: {results['total_time']:.2f}s")
        
        if results['errors']:
            logging.error("\nErrors encountered:")
            for error in results['errors']:
                logging.error(error)
                
    except Exception as e:
        logging.error(f"Pipeline execution failed: {e}", exc_info=True)
    finally:
        logging.info("Pipeline execution completed")
