import torchtune as tt
import transformers

# Load the finetuned model
model = tt.load_checkpoint('llama_finetuned.pth', map_location='cpu')
model_wrapper = tt.ModelWrapper(model)

# Load the tokenizer
tokenizer = transformers.LLaMATokenizer.from_pretrained('llama-base')

def generate_text(input_ids, attention_mask, max_length=512, temperature=1.0):
    model_wrapper.eval()
    output = model_wrapper.generate(input_ids, attention_mask, max_length=max_length, temperature=temperature)
    generated_text = tokenizer.decode(output, skip_special_tokens=True)
    return generated_text

# Define the input prompt or context
input_text = "This is a sample input prompt."

# Tokenize the input
input_ids = tokenizer.encode(input_text, return_tensors='pt')
attention_mask = tokenizer.encode(input_text, return_tensors='pt', max_length=512, truncation=True)

# Generate text
generated_text = generate_text(input_ids, attention_mask)

print(generated_text)