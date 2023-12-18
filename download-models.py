import os

# download the models
os.environ['TRANSFORMERS_CACHE']="./model-storage"
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token = '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')
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", token = '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')

# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-hf", token = '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')
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-hf", token = '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')