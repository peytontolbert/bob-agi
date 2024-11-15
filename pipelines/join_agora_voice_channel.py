from app.models.qwen2vl import Qwen2VL
from app.agents.task_manager import TaskManager, Task
import time
import os

def join_agora_voice_channel(browser):
# Get Discord credentials
    DISCORD_USER = os.getenv('DISCORD_USER')
    DISCORD_PASS = os.getenv('DISCORD_PASS')
    
    qwen2vl = Qwen2VL()
    task_manager = TaskManager(qwen2vl, browser)    
    
    # Define tasks for Discord login
    tasks = [
        Task(
            name="continue_in_browser",
            action="click",
            target="Continue in Browser, small link located below Open App",
            verification="Login textbox is visible"
        ),
        Task(
            name="enter_username",
            action="type",
            target="TEXT BOX located below the email or phone number",
            value=DISCORD_USER,
            verification="Check if username is entered in the text box"
        ),
        Task(
            name="enter_password",
            action="type",
            target="TEXT BOX located below the password field",
            value=DISCORD_PASS,
            verification="Check if password field is visible"
        ),
        Task(
            name="log_in_button",
            action="click",
            target="white text Log in, located inside the blue button",
            verification="Check if Discord dashboard is visible"
        )
    ]

    # Add tasks to manager
    for task in tasks:
        task_manager.add_task(task)

    # Navigate to Discord
    browser.navigate("https://discord.com/channels/@me")
    time.sleep(2)  # Wait for initial page load

    # Run all tasks
    success = task_manager.run_tasks(max_retries=3, delay=2.0)
    
    if success:
        print("All tasks completed successfully!")
        
        # Navigate to specific channel after successful login
        browser.navigate("https://discord.com/channels/999382051935506503/999382052392681605")
        time.sleep(5)
        # Create a new TaskManager instance for the channel tasks
        channel_task_manager = TaskManager(qwen2vl, browser)
        
        join_voice_button_task = Task(
            name="join_voice_button",
            action="click",
            target="white text Join Voice, located inside the green button",
            verification="Joined voice channel"
        )
        
        channel_task_manager.add_task(join_voice_button_task)

        # Run the channel tasks with appropriate delays
        success = channel_task_manager.run_tasks(max_retries=3, delay=5.0)  # Increased delay for page load
        if success:
            print("Successfully joined voice channel!")
        else:
            print("Failed to complete channel tasks")
    else:
        print("Failed to complete login tasks")
