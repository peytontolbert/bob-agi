"""
This is the text Agent that completes text tasks for the input
"""
from app.agents.base import BaseAgent

class TextAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def complete_task(self, input):
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": input}],
            max_tokens=5000,
        )
        print(response.choices[0].message.content)
        return response.choices[0].message.content

