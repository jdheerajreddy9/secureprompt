import os
from dotenv import load_dotenv
from openai import OpenAI
from typing import Dict, Any, Optional
import logging

class LLMInterface:
    def __init__(self, api_key: Optional[str] = None, client=None):
        self._setup_logging()
        load_dotenv()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("No API key provided")
            
        # Allow injection of client for testing
        self.client = client or OpenAI(api_key=self.api_key)
        self.logger.info("Successfully initialized OpenAI client")

    def _setup_logging(self):
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def generate_response(self, prompt: str, max_tokens: int = 150, system_prompt: str = None) -> Dict[str, Any]:
        """Generate a response using the LLM."""
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

        try:
            self.logger.info(f"Generating response for prompt: {prompt[:50]}...")
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = self.client.chat.completions.create(
                model="gpt-4-0613",
                messages=messages,
                max_tokens=max_tokens,
                n=1,
                temperature=0.7,
            )
            return {
                "status": "success",
                "response": response.choices[0].message.content.strip(),
                "usage": response.usage.total_tokens
            }
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }

    def analyze_prompt_safety(self, prompt: str) -> Dict[str, Any]:
        try:
            self.logger.info(f"Analyzing safety for prompt: {prompt[:50]}...")
            response = self.client.moderations.create(input=prompt)
            result = response.results[0]
            return {
                "status": "success",
                "flagged": result.flagged,
                "categories": dict(result.categories),
                "category_scores": dict(result.category_scores)
            }
        except Exception as e:
            self.logger.error(f"Error in safety analysis: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }