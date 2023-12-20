import os

# download the models
os.environ['TRANSFORMERS_CACHE']="./model-storage"
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
except RuntimeError:
    pass

try:
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token = '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')
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", token = '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')
except RuntimeError:
    pass

try:
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-hf", token = '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')
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-hf", token = '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')
except RuntimeError:
    pass