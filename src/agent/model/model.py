import os
import datetime
from src.utils.logger_config import logger

class Model:
    def __init__(self, name=None):
        self.name = name
        
    def predict(self, stop=None):
        raise NotImplementedError   
        
    def log_conversation(self, prompt, response, log_file=None):
        if log_file is None:
            today = datetime.datetime.now().strftime("%Y%m%d")
            log_file = f"logs/conversations_{today}.txt"
        if not os.path.exists(log_file):
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        model_name = self.name or "Unknown Model"
        
        log_content = f"\n--- {timestamp} ({model_name}) ---\n"
        log_content += f"User:\n {prompt}\n"
        log_content += f"Model:\n {response}\n"
        log_content += "-" * 40 + "\n"
        
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log_content)
            
        logger.info(f"Logged conversation to {log_file}")
        
        return log_file 