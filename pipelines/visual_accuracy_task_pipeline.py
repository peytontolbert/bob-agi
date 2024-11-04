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
    element_identifier: str
    action: str  # click, type, wait
    expected_next: str
    timeout: float = 10.0
    retry_count: int = 3

class DiscordVoiceChannelPipeline:
    def __init__(self, vision_agent, eyes, hands):
        self.vision_pipeline = VisualAccuracyPipeline(vision_agent, eyes)
        self.hands = hands
        self.eyes = eyes
        
        # Configure task steps
        self.task_steps = [
            TaskStep(
                description="Find Continue in Browser button",
                element_identifier="Continue in browser link or button",
                action="click",
                expected_next="Discord login page",
                timeout=5.0
            ),
            TaskStep(
                description="Find and click login button",
                element_identifier="Login button",
                action="click", 
                expected_next="Discord main interface",
                timeout=8.0
            ),
            TaskStep(
                description="Locate Agora server",
                element_identifier="Agora server icon or name",
                action="click",
                expected_next="Agora server channels",
                timeout=5.0
            ),
            TaskStep(
                description="Find Voice Channels section",
                element_identifier="Voice Channels heading or section",
                action="wait",
                expected_next="Voice channel list",
                timeout=3.0
            ),
            TaskStep(
                description="Join Agora voice channel",
                element_identifier="Agora voice channel",
                action="click",
                expected_next="Connected to voice",
                timeout=5.0
            )
        ]
        
        # Initialize results tracking
        self.results = []
        self.current_step = 0
        
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

    def _execute_step(self, step: TaskStep) -> Dict[str, Any]:
        """
        Executes a single step in the pipeline with retries.
        
        Args:
            step: TaskStep to execute
            
        Returns:
            Dict containing step execution results
        """
        start_time = time.time()
        
        for attempt in range(step.retry_count):
            try:
                # Find the element
                detection_result = self.vision_pipeline.analyze_scene_and_find_element(
                    step.element_identifier
                )
                
                if not detection_result.get('element_found'):
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
                
                # Execute the action
                if step.action == "click":
                    element_details = detection_result['element_details']
                    click_success = self.hands.click_element(element_details)
                    
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
                
                # Verify expected next state
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
                    'detection_confidence': detection_result.get('confidence', 0)
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
                
        return {
            'success': False,
            'step': step.description,
            'error_message': "Max retries exceeded",
            'time_taken': time.time() - start_time
        }

    def _verify_next_state(self, expected_state: str, timeout: float) -> bool:
        """
        Verifies the next expected state is reached.
        
        Args:
            expected_state: Description of expected state
            timeout: Maximum time to wait for state
            
        Returns:
            bool: True if expected state is reached
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Get current scene understanding
            scene_result = self.vision_pipeline.analyze_scene_and_find_element(expected_state)
            
            if scene_result.get('element_found'):
                return True
                
            time.sleep(0.5)
            
        return False

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
