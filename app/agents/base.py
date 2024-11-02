from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

class BaseAgent():
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)
