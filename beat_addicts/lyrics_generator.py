import torch
torch.set_default_device('cpu')

from transformers import GPT2Tokenizer, GPT2LMHeadModel

class LyricsGenerator:
    def __init__(self, model_name="gpt2"):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to('cpu')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = 150  # CPU-optimized length

    def generate(self, prompt, max_length=None, progress_callback=None):
        inputs = self.tokenizer(prompt, return_tensors="pt").to('cpu')
        
        if progress_callback:
            progress_callback(0.2)  # Encoding done
            
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=max_length or self.max_length,
            temperature=0.7,
            do_sample=True
        )
        
        if progress_callback:
            progress_callback(0.8)  # Generation done
            
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)