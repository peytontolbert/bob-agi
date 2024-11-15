from typing import List, Dict, Optional
from models.qwen2vl import Qwen2VL
from computer.qwen_browser import BrowserController
import time
import re

import json


class MouseControllerHelper:
    def __init__(self, browser: BrowserController, qwen2vl: Qwen2VL):
        self.browser = browser
        self.qwen2vl = qwen2vl

    def parse_coordinates(self, result):
        """Parse the x and y coordinates from the TextAgent result."""
        # Ensure result is a string
        if isinstance(result, list):
            result = ' '.join(result)
        elif not isinstance(result, str):
            print(f"Unexpected result type: {type(result)}")
            return None, None

        # Define regex patterns for different coordinate formats
        patterns = [
            r'\(x:\s*(\d+),\s*y:\s*(\d+)\)',  # Pattern: (x: 488, y: 552)
            r'\((\d+),\s*(\d+)\)'              # Pattern: (488, 552)
        ]

        for pattern in patterns:
            match = re.search(pattern, result)
            if match:
                # Get coordinates in screenshot space (1000x1000)
                screenshot_x, screenshot_y = map(int, match.groups())
                print(f"Found coordinates: ({screenshot_x}, {screenshot_y})")
                return int(screenshot_x), int(screenshot_y)

        print("No valid coordinates found in the result.")
        return None, None

    def locate_element_coordinates(self, element_name):
        """Ask the TextAgent to locate the precise coordinates of an element."""
        self.browser.take_screenshot("images/element_screenshot.png")
        result = self.qwen2vl.chat(input={
            "query": f"Please locate the center coordinates of:\n{element_name}\n reply with the exact coordinates as (x: , y: ) ",
            "image": "images/element_screenshot.png"
        })
        x, y = self.parse_coordinates(result)
        print(f"Located coordinates for '{element_name}': ({x}, {y})")
        return x, y

    def verify_mouse_position(self, viewport_x, viewport_y, element_name):
        """Verify mouse position."""
        self.browser.move_mouse_to(viewport_x, viewport_y)
        filename = f"images/mouse_position_{int(viewport_x)}_{int(viewport_y)}.png"
        self.browser.take_screenshot(filename)
        
        result = self.qwen2vl.chat(input={
            "query": f"""
Is '{element_name}' precisely highlighted with the red circle? 
Reply with a JSON object containing:
- "confidence": a score between 0 and 100
            """,
            "image": filename
        })
        
        try:
            if isinstance(result, list) and len(result) > 0:
                data = json.loads(result[0].strip())
            elif isinstance(result, str):
                data = json.loads(result.strip())
            else:
                print(f"Unexpected result format: {result}")
                return 0.0
            
            return float(data.get("confidence", 0.0))
        except (ValueError, json.JSONDecodeError):
            return 0.0



class Task:
    def __init__(self, name: str, action: str, target: str, value: str = None, verification: str = None):
        self.name = name
        self.action = action  # click, type, move, etc.
        self.target = target  # element to interact with
        self.value = value    # text to type if needed
        self.verification = verification or f"Verify if '{target}' is visible in the image"
        self.completed = False

class TaskManager:
    def __init__(self, qwen2vl: Qwen2VL, browser: BrowserController):
        self.browser = browser
        self.qwen2vl = qwen2vl
        self.tasks: List[Task] = []
        self.current_task_index = 0
        self.verification_prompt = "Does the follow image look like we have completed the first task to move onto the next task? Reply with yes or no."

    def add_task(self, task: Task) -> None:
        """Add a task to the task list."""
        self.tasks.append(task)

    def verify_current_task(self) -> bool:
        """Verify if the current task can be executed by checking the screen."""
        if self.current_task_index >= len(self.tasks):
            return False

        current_task = self.tasks[self.current_task_index]
        screenshot_path = f"images/verification_{current_task.name}.png"
        self.browser.take_screenshot(screenshot_path)

        result = self.qwen2vl.chat(input={
            "query": f"""
Analyze the screenshot and respond with a JSON object containing:
- "visible": true/false indicating if the element is visible
- "confidence": confidence score (0-100)
- "details": additional information about the element's state

Verification task: {current_task.verification}
""",
            "image": screenshot_path
        })

        try:
            if isinstance(result, list):
                result = result[0]
            data = json.loads(result)
            is_visible = data.get("visible", False)
            confidence = data.get("confidence", 0)
            details = data.get("details", "")
            
            print(f"Task '{current_task.name}' verification:")
            print(f"Visible: {is_visible}, Confidence: {confidence}")
            print(f"Details: {details}")
            
            return is_visible and confidence >= 90
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing verification result: {e}")
            return False

    def execute_current_task(self) -> bool:
        """Execute the current task."""
        if self.current_task_index >= len(self.tasks):
            return False

        current_task = self.tasks[self.current_task_index]
        success = False

        try:
            if current_task.action == "click":
                success = click_element(self.browser, self.qwen2vl, current_task.target)
            elif current_task.action == "type":
                success = click_and_type_element(
                    self.browser, 
                    self.qwen2vl, 
                    current_task.target, 
                    current_task.value
                )
            elif current_task.action == "move":
                success = move_to_element(self.browser, self.qwen2vl, current_task.target)
            
            if success:
                print(f"Task '{current_task.name}' executed successfully")
                return True
            else:
                print(f"Failed to execute task '{current_task.name}'")
                return False
        except Exception as e:
            print(f"Error executing task '{current_task.name}': {e}")
            return False

    def run_tasks(self, max_retries: int = 3, delay: float = 2.0) -> bool:
        """Run all tasks in sequence with verification and retry logic."""
        while self.current_task_index < len(self.tasks):
            current_task = self.tasks[self.current_task_index]
            retries = 0
            
            while retries < max_retries:
                print(f"\nAttempting task: {current_task.name} (Attempt {retries + 1}/{max_retries})")
                
                # Execute the task
                if self.execute_current_task():
                    # Wait for any animations or page transitions
                    print(f"Task action executed, waiting {delay} seconds before verification...")
                    time.sleep(delay)
                    
                    # Verify task completion
                    if self.verify_task_completion():
                        print(f"Task '{current_task.name}' completed and verified successfully")
                        current_task.completed = True
                        self.current_task_index += 1
                        break
                    else:
                        print(f"Task completion verification failed, may retry...")
                else:
                    print(f"Task action execution failed")
                
                retries += 1
                if retries < max_retries:
                    print(f"Retrying task '{current_task.name}' in {delay} seconds...")
                    time.sleep(delay)
            
            if retries == max_retries and not current_task.completed:
                print(f"Failed to complete task '{current_task.name}' after {max_retries} attempts")
                return False
        
        return True

    def verify_task_completion(self, screenshot_path="images/task_verification.png"):
        """Verify if the current task has been completed successfully."""
        if self.current_task_index >= len(self.tasks):
            return False
        
        current_task = self.tasks[self.current_task_index]
        
        # Take a fresh screenshot for verification after a short delay
        time.sleep(2)  # Allow time for any UI updates
        print("Taking fresh screenshot for task completion verification...")
        self.browser.take_screenshot(screenshot_path)
        
        result = self.qwen2vl.chat(input={
            "query": f"""
Analyze if the following task has been completed successfully:
{current_task.verification}

Look for these indicators of completion:
1. Expected changes in the page layout
2. New elements that should appear
3. Old elements that should disappear
4. Any success messages or confirmations

Reply with a JSON object containing:
- "completed": true/false
- "confidence": 0-100
- "details": specific observations about the task completion state
""",
            "image": screenshot_path
        })
        
        try:
            if isinstance(result, list):
                result = result[0]
            
            data = json.loads(result)
            is_completed = data.get("completed", False)
            confidence = data.get("confidence", 0)
            details = data.get("details", "")
            
            print(f"\nTask Completion Verification for '{current_task.name}':")
            print(f"Completed: {is_completed}, Confidence: {confidence}")
            print(f"Details: {details}")
            
            # More lenient threshold for task completion
            if is_completed and confidence >= 75:
                return True
            else:
                print(f"Task completion verification failed: {details}")
                return False
            
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing task completion verification result: {e}")
            print(f"Raw result: {result}")
            return False

    def wait_and_verify(self, timeout=10, interval=2):
        """Wait and verify task completion with timeout."""
        if self.current_task_index >= len(self.tasks):
            return False
        
        current_task = self.tasks[self.current_task_index]
        print(f"\nWaiting for task completion: {current_task.name}")
        print(f"Verification criteria: {current_task.verification}")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.verify_task_completion():
                print(f"Task '{current_task.name}' verified successfully")
                return True
            print(f"Verification failed, retrying in {interval} seconds...")
            time.sleep(interval)
        
        print(f"Task '{current_task.name}' verification timed out after {timeout} seconds")
        return False



def click_and_type_element(browser, text_agent, element_name, text_to_type, max_attempts=3):
    """Click an element and type text into it with retries."""
    helper = MouseControllerHelper(browser, text_agent)
    
    for attempt in range(max_attempts):
        print(f"Attempt {attempt + 1}/{max_attempts} to click and type into '{element_name}'")
        
        x, y = helper.locate_element_coordinates(element_name)
        if x is None or y is None:
            print(f"Could not locate coordinates, retrying...")
            continue

        #viewport_x, viewport_y = browser.normalize_coordinates(x, y, from_screenshot=True)
        confidence = helper.verify_mouse_position(x, y, element_name)
        
        if confidence >= 90:
            browser.click_and_type(x, y, text_to_type)
            print(f"Successfully clicked and typed into '{element_name}' at ({x}, {y})")
            return True
        
        print(f"Verification failed with confidence {confidence}, retrying...")
        time.sleep(1)
    
    print(f"Failed to click and type into '{element_name}' after {max_attempts} attempts")
    return False

def click_element(browser, text_agent, element_name, max_attempts=3):
    """Click an element with retries."""
    helper = MouseControllerHelper(browser, text_agent)
    
    for attempt in range(max_attempts):
        print(f"Attempt {attempt + 1}/{max_attempts} to click '{element_name}'")
        
        x, y = helper.locate_element_coordinates(element_name)
        if x is None or y is None:
            print(f"Could not locate coordinates, retrying...")
            continue

        #viewport_x, viewport_y = browser.normalize_coordinates(x, y, from_screenshot=True)
        confidence = helper.verify_mouse_position(x, y, element_name)
        
        if confidence >= 90:
            browser.click_at(x, y)
            print(f"Successfully clicked '{element_name}' at ({x}, {y})")
            return True
        
        print(f"Verification failed with confidence {confidence}, retrying...")
        time.sleep(1)
    
    print(f"Failed to click '{element_name}' after {max_attempts} attempts")
    return False

def move_to_element(browser, text_agent, element_name, max_attempts=3):
    """Move to an element with retries."""
    helper = MouseControllerHelper(browser, text_agent)
    
    for attempt in range(max_attempts):
        print(f"Attempt {attempt + 1}/{max_attempts} to move to '{element_name}'")
        
        x, y = helper.locate_element_coordinates(element_name)
        if x is None or y is None:
            print(f"Could not locate coordinates, retrying...")
            continue

        viewport_x, viewport_y = browser.normalize_coordinates(x, y, from_screenshot=True)
        confidence = helper.verify_mouse_position(viewport_x, viewport_y, element_name)
        
        if confidence >= 90:
            browser.move_mouse_to(viewport_x, viewport_y)
            print(f"Successfully moved to '{element_name}' at ({viewport_x}, {viewport_y})")
            return True
        
        print(f"Verification failed with confidence {confidence}, retrying...")
        time.sleep(1)
    
    print(f"Failed to move to '{element_name}' after {max_attempts} attempts")
    return False







