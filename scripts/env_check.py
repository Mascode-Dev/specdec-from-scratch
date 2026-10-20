import torch
import transformers

# Display torch version, gpu, transfomers version
print(f"Torch version: {torch.__version__}")
print(f"GPU available: {torch.cuda.is_available()}")
print(f"Transformers version: {transformers.__version__}")

# Load a pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
model = transformers.GPT2LMHeadModel.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda")
tokenizer = transformers.GPT2TokenizerFast.from_pretrained(model_name)

# Forward pass on a tokenized input
prompt = "Hello, how are you?"
inputs = tokenizer(prompt, return_tensors="pt")

inputs = inputs.to("cuda")
# Forward pass with no gradient calculation
with torch.inference_mode():
    outputs = model(**inputs)

# Display ids.shape, logits.shape, and the first 5 logits
print(f"ids.shape: {inputs['input_ids'].shape}")
print(f"logits.shape: {outputs.logits.shape}")
print(f"Logits dtype: {outputs.logits.dtype}")
print(f"Logits device: {outputs.logits.device}")

# Verify the shift of logits to the GPU
# Display argmax(logits[0,-1]) and [0,0]
print(f"Argmax logits[0,-1]: {torch.argmax(outputs.logits[0,-1])}")
print(f"Argmax logits[0,0]: {torch.argmax(outputs.logits[0,0])}")