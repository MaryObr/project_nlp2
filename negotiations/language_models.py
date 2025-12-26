# 
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

class HuggingFaceModel:
    def __init__(self, model_name="microsoft/DialoGPT-small", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(device)
        
        # Для использования через pipeline
        self.generator = pipeline(
            "text-generation",
            model=model_name,
            tokenizer=self.tokenizer,
            device=0 if device == "cuda" else -1
        )
    
    def predict_token(self, prompt, max_length=200, temperature=0.7, top_p=0.9):
        """Генерирует текст на основе промпта"""
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Убираем промпт из ответа
        response = response[len(prompt):].strip()
        return response
    
    def vocabulary(self):
        """Возвращает словарь токенов"""
        return list(self.tokenizer.get_vocab().keys())
