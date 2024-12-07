import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, Any, Tuple
import logging
from .llm_interface import LLMInterface

class SecureLLMPipeline:
    def __init__(self, model_path: str = "models/advglue_model.pt"):
        self.logger = logging.getLogger(__name__)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        try:
            # Initialize LLM interface
            self.llm = LLMInterface()
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            self.model = AutoModelForSequenceClassification.from_pretrained(
                'bert-base-uncased',
                num_labels=3
            )
            
            # Try to load trained weights if they exist
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device)['model_state_dict'])
            except FileNotFoundError:
                self.logger.warning(f"No trained model found at {model_path}. Using base model.")
            
            self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            self.logger.error(f"Error initializing pipeline: {str(e)}")
            raise

    def check_prompt_safety(self, prompt: str) -> Dict[str, float]:
        """Check if prompt is potentially adversarial"""
        try:
            # Tokenize
            inputs = self.tokenizer(
                prompt,
                truncation=True,
                max_length=512,
                padding='max_length',
                return_tensors='pt'
            ).to(self.device)

            # Get prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)

            return {
                'safe_score': probs[0][0].item(),
                'adversarial_score': probs[0][1].item(),
                'is_safe': probs[0][0].item() > 0.5
            }
        except Exception as e:
            self.logger.error(f"Error checking prompt safety: {str(e)}")
            return {'is_safe': False, 'safe_score': 0.0, 'adversarial_score': 1.0}

    def generate_safe_alternative(self, prompt: str, safety_scores: Dict[str, float]) -> str:
        """Generate a safer alternative for adversarial prompts"""
        try:
            system_prompt = """
            This prompt may be adversarial. Please generate a safe alternative that:
            1. Preserves the core intent
            2. Removes potential harmful elements
            3. Maintains appropriate language
            Original prompt: {prompt}
            Safe alternative:
            """.format(prompt=prompt)

            response = self.llm.generate_response(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=100
            )
            return response.get('response', prompt)
        except Exception as e:
            self.logger.error(f"Error generating safe alternative: {str(e)}")
            return prompt

    def process_prompt(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        """Process prompt through safety checks before sending to LLM"""
        try:
            # Check prompt safety
            safety_check = self.check_prompt_safety(prompt)
            
            # If prompt is adversarial, generate safe alternative
            if not safety_check['is_safe']:
                safe_prompt = self.generate_safe_alternative(prompt, safety_check)
                self.logger.warning(f"Original prompt modified for safety. Score: {safety_check['adversarial_score']:.2f}")
            else:
                safe_prompt = prompt
                self.logger.info(f"Prompt passed safety check. Score: {safety_check['safe_score']:.2f}")

            # Get LLM response
            response = self.llm.generate_response(safe_prompt)

            return response.get('response', ''), {
                'original_prompt': prompt,
                'used_prompt': safe_prompt,
                'was_modified': prompt != safe_prompt,
                'safety_scores': safety_check
            }

        except Exception as e:
            self.logger.error(f"Error in prompt processing: {str(e)}")
            return "", {"error": str(e)}