import torch
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline, TrainingArguments
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

dataset = load_dataset("beomi/KoAlpaca-v1.1a")
print(dataset)