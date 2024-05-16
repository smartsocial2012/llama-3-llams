import os, torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig, 
    pipeline, 
    Trainer, 
    TrainingArguments)
from accelerate import Accelerator
accelerator = Accelerator()

torch.cuda.empty_cache()
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

BASE_MODEL = 'meta-llama/Meta-Llama-3-8B-Instruct'
# BASE_MODEL = 'BM-K/KoChatBART'

dataset = load_dataset("beomi/KoAlpaca-v1.1a", split='train')
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto")

tokenizer.add_special_tokens({'pad_token': '[PAD]'})

messages = [
    {"role": "system", "content": "You are 'Llama', a helpful assistant. Reply in 한국어 only."},
    {"role": "user", "content": "니 한국어 잘하나?"},
]

def preprocess_function(examples):
    instructions = examples['instruction']
    responses = examples['output']
    return tokenizer.prepare_seq2seq_batch(
        src_texts=instructions,
        tgt_texts=responses,
        return_tensors='pt',
        max_length=1024,  # Adjust the max length as needed
        truncation=True  # Enable truncation
    )

tokenized_dataset = dataset.map(preprocess_function, batched=True, batch_size=2)

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = model.generate(
    input_ids,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
response = outputs[0][input_ids.shape[-1]:]
print(tokenizer.decode(response, skip_special_tokens=True))

training_args = TrainingArguments(
    output_dir="./finetuned_model",
    num_train_epochs=3,           # Adjust based on your needs
    per_device_train_batch_size=1,  # Adjust based on your hardware
    save_steps=10000,             # Adjust based on your preferences
    eval_steps=5000,              # Adjust based on your preferences
    # Add additional arguments like learning rate, weight decay, etc.
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    # eval_dataset=tokenized_dataset["validation"],
    # Add data collator if needed
)

# Start training
trainer.train()

# Save the fine-tuned model
trainer.save_model("./finetuned_model")
