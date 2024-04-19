from datetime import datetime as dt
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import pipeline as ple

import gc
import torch

class QueryParser():
    def __init__(self, model_path):
        """This function constructs the Keyword Extractor model.
        
        PARAMETERS
        model_path - The path to load the Keyword extractor model.
        """
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False
        )
        self.model = None
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path + "tokenizer",
            trust_remote_code=True)
        self.tokenizer.padding_side = 'right'
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.tokenizer.add_eos_token = True

    def apply_chat_template(self, input_str):
        """Applies the prompt from the input string to the model.
        
        PARAMETER
        input_str - The string of the user's input.
        """


        prompt = ("<|system|>\nYou are a chatbot " + 
                  "that assists in providing " + 
                  "information to the user. " +
                  "Please generate a keyword " + 
                  "from the user's question.</s>\n<|user|>\n")
        prompt+= input_str
        prompt += "</s>\n<|assistant|>\n"
        return prompt
    
    def generate_query(self, input_str):
        """
            Generates and returns the search query by the Keyword Generator
            model.

            PARAMETERS
            input_str - The string to generate query from.
        """


        self.model = AutoModelForCausalLM.from_pretrained(self.model_path,
            quantization_config=self.bnb_config)
        prompt = self.apply_chat_template(input_str)
        pipe = ple("text-generation", model=self.model, tokenizer=self.tokenizer,
            torch_dtype=torch.bfloat16, device_map="auto")
        query = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, 
                    top_k=50, top_p=0.95)
        query = query.split("<|assistant|>\n")
        query = query[1]
        query = query.split("<\s>")
        query = query[0]
        self.model = None
        pipe = None
        del self.model, pipe
        gc.collect()
        torch.cuda.empty_cache()
        return query