"""
Production-grade pipeline for UI automation with robust error handling and monitoring.
"""
import logging
import time
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import base64
from pathlib import Path
import json
import traceback
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, TimeoutError

from pipelines.visual_accuracy_pipeline import VisualAccuracyPipeline, DetectionResult

@dataclass
class TaskStep:
    """Represents a step in the task pipeline"""
    id: str  # Unique identifier for the step
    description: str
    element_identifiers: List[str]
    visual_cues: List[str]
    action: str
    expected_next: List[str]
    timeout: float = 10.0
    retry_count: int = 3
    recovery_steps: Optional[List[str]] = None  # Steps to take if this step fails
    validation_checks: Optional[List[str]] = None  # Additional validation checks

@dataclass
class PipelineMetrics:
    """Tracks pipeline execution metrics"""
    start_time: float
    steps_completed: int = 0
    total_attempts: int = 0
    successful_attempts: int = 0
    failed_attempts: int = 0
    detection_times: List[float] = None
    action_times: List[float] = None
    errors: List[Dict] = None
    
    def __post_init__(self):
        self.detection_times = []
        self.action_times = []
        self.errors = []
    
    def add_detection_time(self, time_taken: float):
        self.detection_times.append(time_taken)
    
    def add_action_time(self, time_taken: float):
        self.action_times.append(time_taken)
    
    def add_error(self, step_id: str, error_type: str, error_msg: str):
        self.errors.append({
            'step_id': step_id,
            'type': error_type,
            'message': error_msg,
            'timestamp': time.time()
        })

class DiscordVoiceChannelPipeline:
    def __init__(self, vision_agent, eyes, hands, text_agent, config: Optional[Dict] = None):
        self.vision_pipeline = VisualAccuracyPipeline(vision_agent, eyes)
        self.hands = hands
        self.eyes = eyes
        self.text_agent = text_agent
        self.config = config or self._get_default_config()
        
        # Initialize metrics
        self.metrics = PipelineMetrics(start_time=time.time())
        
        # Setup logging
        self._setup_logging()
        
        # Initialize task steps
        self.task_steps = self._initialize_task_steps()
        
        # State management
        self.current_step = 0
        self.pipeline_state = 'initialized'
        
        # Debug and monitoring
        self.debug_path = Path(self.config['debug_path'])
        self.debug_path.mkdir(parents=True, exist_ok=True)

    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'debug_path': 'debug_output/element_detection',
            'max_retries': 3,
            'timeout': 30,
            'action_delay': 0.5,
            'verification_delay': 1.0,
            'logging_level': logging.INFO,
            'save_debug_images': True,
            'metrics_enabled': True
        }

    def _setup_logging(self):
        """Configure logging for the pipeline"""
        log_path = Path('logs/pipeline')
        log_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_path / f"pipeline_{timestamp}.log"
        
        logging.basicConfig(
            level=self.config['logging_level'],
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

    @contextmanager
    def step_execution_context(self, step: TaskStep):
        """Context manager for step execution with timing and error handling"""
        step_start = time.time()
        step_metrics = {'start_time': step_start, 'attempts': 0}
        
        try:
            yield step_metrics
        except Exception as e:
            self.metrics.add_error(step.id, 'execution_error', str(e))
            logging.error(f"Step {step.id} failed: {e}")
            raise
        finally:
            step_metrics['duration'] = time.time() - step_start
            self._save_step_metrics(step.id, step_metrics)

    def execute_pipeline(self) -> Dict[str, Any]:
        """Execute the pipeline with comprehensive error handling and metrics"""
        pipeline_start_time = time.time()
        self.pipeline_state = 'running'
        results = {
            'success': False,
            'steps_completed': 0,
            'total_steps': len(self.task_steps),
            'metrics': {},
            'errors': [],
            'start_time': pipeline_start_time
        }

        try:
            with ThreadPoolExecutor() as executor:
                for step in self.task_steps:
                    if self.pipeline_state != 'running':
                        break

                    with self.step_execution_context(step) as metrics:
                        future = executor.submit(self._execute_step, step)
                        try:
                            step_result = future.result(timeout=step.timeout)
                            if not step_result['success']:
                                if not self._handle_step_failure(step, step_result):
                                    results['errors'].append({
                                        'step': step.id,
                                        'error': step_result.get('error', 'Unknown error'),
                                        'time': time.time()
                                    })
                                    break
                            results['steps_completed'] += 1
                            metrics['success'] = True
                        except TimeoutError:
                            error_msg = f"Step timed out after {step.timeout}s"
                            self.metrics.add_error(step.id, 'timeout', error_msg)
                            results['errors'].append({
                                'step': step.id,
                                'error': error_msg,
                                'time': time.time()
                            })
                            break

            # Calculate total execution time
            total_time = time.time() - pipeline_start_time
            
            # Update final results
            results.update({
                'success': (results['steps_completed'] == len(self.task_steps)),
                'metrics': self._collect_metrics(),
                'total_time': total_time,
                'end_time': time.time()
            })
            
        except Exception as e:
            error_msg = str(e)
            self.metrics.add_error('pipeline', 'execution_error', error_msg)
            results['errors'].append({
                'step': 'pipeline',
                'error': error_msg,
                'traceback': traceback.format_exc(),
                'time': time.time()
            })
            logging.error(f"Pipeline execution failed: {traceback.format_exc()}")
            
        finally:
            self.pipeline_state = 'completed'
            if 'total_time' not in results:
                results['total_time'] = time.time() - pipeline_start_time
            self._save_pipeline_results(results)
            
        return results

    def _handle_step_failure(self, step: TaskStep, result: Dict) -> bool:
        """Handle step failure with recovery attempts"""
        if not step.recovery_steps:
            return False
            
        for recovery_step in step.recovery_steps:
            try:
                logging.info(f"Attempting recovery step: {recovery_step}")
                # Execute recovery logic
                if self._execute_recovery_step(recovery_step):
                    return True
            except Exception as e:
                self.metrics.add_error(step.id, 'recovery_error', str(e))
                
        return False

    def _execute_recovery_step(self, recovery_step: str) -> bool:
        """Execute a recovery step"""
        # Implement recovery logic based on step type
        return False  # Default to failed recovery

    def _collect_metrics(self) -> Dict[str, Any]:
        """Collect and format pipeline metrics"""
        return {
            'duration': time.time() - self.metrics.start_time,
            'steps_completed': self.metrics.steps_completed,
            'success_rate': self.metrics.successful_attempts / max(1, self.metrics.total_attempts),
            'avg_detection_time': sum(self.metrics.detection_times) / max(1, len(self.metrics.detection_times)),
            'avg_action_time': sum(self.metrics.action_times) / max(1, len(self.metrics.action_times)),
            'error_count': len(self.metrics.errors),
            'error_types': self._categorize_errors(),
            'total_attempts': self.metrics.total_attempts,
            'successful_attempts': self.metrics.successful_attempts,
            'failed_attempts': self.metrics.failed_attempts
        }

    def _categorize_errors(self) -> Dict[str, int]:
        """Categorize errors by type"""
        error_counts = {}
        for error in self.metrics.errors:
            error_counts[error['type']] = error_counts.get(error['type'], 0) + 1
        return error_counts

    def _save_step_metrics(self, step_id: str, metrics: Dict):
        """Save metrics for individual step"""
        metrics_path = self.debug_path / 'metrics'
        metrics_path.mkdir(exist_ok=True)
        
        with open(metrics_path / f"step_{step_id}_{int(time.time())}.json", 'w') as f:
            json.dump(metrics, f, indent=2)

    def pause_pipeline(self):
        """Pause pipeline execution"""
        self.pipeline_state = 'paused'
        logging.info("Pipeline execution paused")

    def resume_pipeline(self):
        """Resume pipeline execution"""
        self.pipeline_state = 'running'
        logging.info("Pipeline execution resumed")

    def get_pipeline_state(self) -> Dict[str, Any]:
        """Get current pipeline state and metrics"""
        return {
            'state': self.pipeline_state,
            'current_step': self.current_step,
            'metrics': self._collect_metrics(),
            'last_error': self.metrics.errors[-1] if self.metrics.errors else None
        }

    def _initialize_task_steps(self) -> List[TaskStep]:
        """Initialize task steps for Discord voice channel pipeline"""
        return [
            TaskStep(
                id="continue_browser",
                description="Find Continue in Browser button",
                element_identifiers=[
                    "Continue in Browser",
                    "Continue in browser",
                    "Use Discord in browser"
                ],
                visual_cues=[
                    "blue button",
                    "prominent button",
                    "center button"
                ],
                action="click",
                expected_next=[
                    "Discord login page",
                    "Login with Discord"
                ],
                timeout=8.0,
                retry_count=3,
                recovery_steps=[
                    "refresh_page",
                    "clear_cookies"
                ],
                validation_checks=[
                    "button_visible",
                    "button_clickable"
                ]
            ),
            TaskStep(
                id="login_button",
                description="Find and click login button",
                element_identifiers=[
                    "Login",
                    "Log In",
                    "Sign In"
                ],
                visual_cues=[
                    "blue button",
                    "login form button"
                ],
                action="click",
                expected_next=[
                    "Discord main interface",
                    "Server list"
                ],
                timeout=10.0
            ),
            TaskStep(
                id="find_server",
                description="Locate Agora server",
                element_identifiers=[
                    "Agora",
                    "Agora server",
                    "Agora Discord"
                ],
                visual_cues=[
                    "server icon",
                    "left sidebar"
                ],
                action="click",
                expected_next=[
                    "Agora server channels",
                    "Voice channels list"
                ],
                timeout=5.0
            ),
            TaskStep(
                id="find_voice_section",
                description="Find Voice Channels section",
                element_identifiers=[
                    "Voice Channels",
                    "VOICE CHANNELS"
                ],
                visual_cues=[
                    "channel category",
                    "voice section"
                ],
                action="wait",
                expected_next=[
                    "Voice channel list"
                ],
                timeout=3.0
            ),
            TaskStep(
                id="join_voice",
                description="Join Agora voice channel",
                element_identifiers=[
                    "Agora Voice",
                    "General Voice"
                ],
                visual_cues=[
                    "speaker icon",
                    "voice channel"
                ],
                action="click",
                expected_next=[
                    "Connected to voice",
                    "In voice channel"
                ],
                timeout=5.0
            )
        ]

    def _execute_step(self, step: TaskStep) -> Dict[str, Any]:
        """Execute a single pipeline step with enhanced error handling and validation"""
        start_time = time.time()
        self.metrics.total_attempts += 1
        
        try:
            # Pre-step validation
            if not self._validate_step_preconditions(step):
                raise ValueError(f"Precondition check failed for step {step.id}")

            # Find element using vision pipeline
            detection_start = time.time()
            element = self._find_element_with_retries(step)
            self.metrics.add_detection_time(time.time() - detection_start)

            if not element:
                self.metrics.failed_attempts += 1
                return {
                    'success': False,
                    'step': step.description,
                    'error': 'Element not found',
                    'time_taken': time.time() - start_time
                }

            # Execute action based on step type
            action_start = time.time()
            
            # Create element dict with required fields for hands
            element_dict = {
                'coordinates': element['coordinates'],
                'type': 'ui_element'
            }
            
            action_success = self._execute_action(step, element_dict)
            self.metrics.add_action_time(time.time() - action_start)

            if not action_success:
                self.metrics.failed_attempts += 1
                return {
                    'success': False,
                    'step': step.description,
                    'error': 'Action failed',
                    'time_taken': time.time() - start_time
                }

            # Verify step completion
            if not self._verify_step_completion(step):
                self.metrics.failed_attempts += 1
                return {
                    'success': False,
                    'step': step.description,
                    'error': 'Step verification failed',
                    'time_taken': time.time() - start_time
                }

            self.metrics.successful_attempts += 1
            return {
                'success': True,
                'step': step.description,
                'time_taken': time.time() - start_time
            }

        except Exception as e:
            self.metrics.failed_attempts += 1
            self.metrics.add_error(step.id, 'execution_error', str(e))
            logging.error(f"Step execution error: {traceback.format_exc()}")
            return {
                'success': False,
                'step': step.description,
                'error': str(e),
                'time_taken': time.time() - start_time
            }

    def _find_element_with_retries(self, step: TaskStep) -> Optional[Dict]:
        """Find element with retries"""
        for attempt in range(step.retry_count):
            try:
                # Generate detection prompt
                prompt = f"Find {step.element_identifiers[0]} return coordinates"
                logging.info(f"Attempting detection with prompt: {prompt}")
                
                # Get screen image using correct method
                screen_image = self.eyes.get_screen_image()
                if screen_image is None:
                    logging.error("Failed to capture screen image")
                    continue
                
                # Get detection result
                result = self.vision_pipeline.vision_agent.detect_ui_elements(
                    screen_image,
                    query=step.element_identifiers[0]
                )
                
                if result['status'] == 'success' and result['detections']:
                    element = result['detections'][0]
                    logging.info(f"Element found: {element['coordinates']}")
                    return element
                    
                if attempt < step.retry_count - 1:
                    logging.info(f"Retrying detection (attempt {attempt + 1}/{step.retry_count})")
                    time.sleep(self.config['action_delay'])
                    
            except Exception as e:
                logging.warning(f"Detection error on attempt {attempt + 1}: {e}")
                if attempt == step.retry_count - 1:
                    raise
                    
        return None

    def _validate_step_preconditions(self, step: TaskStep) -> bool:
        """Validate step preconditions"""
        if not step.validation_checks:
            return True
            
        for check in step.validation_checks:
            if not self._run_validation_check(check):
                logging.error(f"Validation check failed: {check}")
                return False
        return True

    def _run_validation_check(self, check: str) -> bool:
        """Run a specific validation check"""
        validation_map = {
            'button_visible': self._check_button_visible,
            'button_clickable': self._check_button_clickable,
            # Add more validation checks as needed
        }
        
        if check in validation_map:
            return validation_map[check]()
        return True

    def _check_button_visible(self) -> bool:
        """Check if button is visible in current screen"""
        # Implement visibility check
        return True

    def _check_button_clickable(self) -> bool:
        """Check if button is clickable"""
        # Implement clickability check
        return True

    def _execute_action(self, step: TaskStep, element: Dict) -> bool:
        """Execute step action with proper handling"""
        try:
            if step.action == "click":
                return self.hands.click_element(element)
            elif step.action == "wait":
                time.sleep(self.config['action_delay'])
                return True
            # Add more action types as needed
            return False
        except Exception as e:
            logging.error(f"Action execution error: {e}")
            return False

    def _verify_step_completion(self, step: TaskStep, timeout: float = None) -> bool:
        """Verify step completion"""
        timeout = timeout or step.timeout
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                # Check for any expected next elements
                for expected in step.expected_next:
                    result = self.vision_pipeline.vision_agent.detect_ui_elements(
                        self.eyes.get_screen_image(),
                        query=expected
                    )
                    
                    if result['status'] == 'success' and result['detections']:
                        logging.info(f"Verification successful: found {expected}")
                        return True
                        
                time.sleep(0.5)  # Brief delay between checks
                
            except Exception as e:
                logging.warning(f"Verification check failed: {e}")
                continue
        
        logging.error(f"Step verification timed out after {timeout}s")
        return False

    def _save_pipeline_results(self, results: Dict[str, Any]):
        """Save pipeline execution results"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_path = self.debug_path / f"pipeline_results_{timestamp}.json"
            
            # Convert results to serializable format
            serializable_results = {
                'success': results['success'],
                'steps_completed': results['steps_completed'],
                'total_steps': results['total_steps'],
                'total_time': results['total_time'],
                'metrics': results.get('metrics', {}),
                'errors': [
                    {
                        'step': error.get('step', 'unknown'),
                        'error': str(error.get('error', '')),
                        'time': error.get('time', 0)
                    }
                    for error in results.get('errors', [])
                ],
                'execution_timeline': {
                    'start_time': results.get('start_time', 0),
                    'end_time': results.get('end_time', time.time())
                }
            }
            
            with open(results_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)
                
            logging.info(f"Saved pipeline results to {results_path}")
            
        except Exception as e:
            logging.error(f"Failed to save pipeline results: {e}")

if __name__ == "__main__":
    from app.agents.vision import VisionAgent
    from app.env.screen import Screen
    from app.env.computer import Computer
    from app.env.senses.eyesight import Eyesight
    from app.actions.hands import Hands
    from app.agents.text import TextAgent
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
        text_agent = TextAgent()
        
        # Create pipeline with text agent
        pipeline = DiscordVoiceChannelPipeline(vision_agent, eyes, hands, text_agent)
        
        # Execute pipeline
        logging.info("Starting Discord voice channel pipeline...")
        results = pipeline.execute_pipeline()
        
        # Log results
        logging.info("\nPipeline Results:")
        logging.info(f"Success: {results['success']}")
        logging.info(f"Steps completed: {results['steps_completed']}/{results['total_steps']}")
        logging.info(f"Total time: {results.get('total_time', 0):.2f}s")
        
        if results.get('errors'):
            logging.error("\nErrors encountered:")
            for error in results['errors']:
                logging.error(f"Step: {error.get('step', 'unknown')} - {error.get('error', 'Unknown error')}")
                
    except Exception as e:
        logging.error(f"Pipeline execution failed: {e}", exc_info=True)
    finally:
        logging.info("Pipeline execution completed")
