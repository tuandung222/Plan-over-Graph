import os
import torch
from transformers import pipeline
from huggingface_hub import login
from src.agent.model.model import Model
from src.utils.logger_config import logger, COLOR_CODES, RESET

class LlamaWrapper(Model):
    def __init__(self, model = "meta-llama/Llama-3.1-8B-Instruct"):
        super().__init__(name="LlamaWrapper")
        self.model = model

    def predict(self, prompt, max_new_tokens=32768, stop=None):        
        pipe = pipeline(
            "text-generation",
            model=self.model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        messages = [
            {"role": "user", "content": prompt},
        ]        
        try:
            outputs = pipe(
                messages,
                max_new_tokens=max_new_tokens,
                pad_token_id=pipe.tokenizer.eos_token_id,
                temperature=0.2,
            )
            response_text = outputs[0]["generated_text"][-1]['content']
        except Exception as e:            
            print(e)
            logger.error(f"Error: {COLOR_CODES['RED']}{e}{RESET}")
            
            exit(1)

        if "deepseek-r1" in self.model.lower():
            import re
            response_text = re.sub(r'<think>.*?<\/think>', '', response_text, flags=re.DOTALL)
        return response_text

def main():
    llama_wrapper = LlamaWrapper(model="/data/share/data/llama-factory/1/DeepSeek-R1-Distill-Llama-8B")
    prompt = "What is the capital of France?"
    response = llama_wrapper.predict(prompt)
    print(response)

if __name__ == "__main__":
    main()