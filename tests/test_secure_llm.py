import logging
from datetime import datetime
from src.secure_llm_pipeline import SecureLLMPipeline
from pathlib import Path
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def save_interaction(prompt: str, response: str, metadata: dict, output_file: str = "interactions.json"):
    """Save interaction history"""
    interaction = {
        "prompt": prompt,
        "response": response,
        "metadata": metadata,
        "timestamp": str(datetime.now())
    }
    
    try:
        if Path(output_file).exists():
            with open(output_file, 'r') as f:
                data = json.load(f)
        else:
            data = []
            
        data.append(interaction)
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving interaction: {str(e)}")

def main():
    try:
        # Initialize pipeline
        logger.info("Initializing Secure LLM Pipeline...")
        pipeline = SecureLLMPipeline()
        
        # Interactive loop
        while True:
            print("\nSecure LLM Interface")
            print("Enter 'quit' to exit")
            print("-" * 50)
            
            prompt = input("\nEnter your prompt: ").strip()
            
            if prompt.lower() == 'quit':
                break
                
            if not prompt:
                print("Please enter a prompt.")
                continue

            try:
                # Process prompt
                response, metadata = pipeline.process_prompt(prompt)
                
                # Print results
                print("\nResults:")
                print("-" * 50)
                if metadata.get('was_modified'):
                    print("\n⚠️  Prompt was modified for safety:")
                    print(f"Original: {prompt}")
                    print(f"Modified: {metadata['used_prompt']}")
                else:
                    print("\n✅ Prompt passed safety checks")
                
                print("\nSafety Scores:")
                scores = metadata.get('safety_scores', {})
                print(f"Safe Score: {scores.get('safe_score', 0):.2f}")
                print(f"Adversarial Score: {scores.get('adversarial_score', 0):.2f}")
                
                print("\nLLM Response:")
                print("-" * 50)
                print(response)
                
                # Save interaction
                save_interaction(prompt, response, metadata)
                
            except Exception as e:
                logger.error(f"Error processing prompt: {str(e)}")
                print("An error occurred while processing your prompt. Please try again.")

    except Exception as e:
        logger.error(f"Pipeline initialization failed: {str(e)}")
        return

if __name__ == "__main__":
    main()