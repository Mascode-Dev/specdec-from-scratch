import time
import torch
import transformers

model = transformers.GPT2LMHeadModel.from_pretrained("gpt2", torch_dtype=torch.float16).to("cuda")
tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2")
model.eval()

prompt = "Hello, how are you? I'm fine, thanks for"
enc = tokenizer(prompt, return_tensors="pt").to("cuda")
input_ids = enc["input_ids"]
attn = enc["attention_mask"]
prompt_len = input_ids.shape[1]

MAX_NEW = 200
generated = []

# Add KV-Cache
past_key_values = None
temperature = 1.3

torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize(); t0 = time.time()
with torch.inference_mode():
    out = model(input_ids=input_ids, attention_mask=attn, use_cache=True)
    past = out.past_key_values

    # each step take next_id, then forward only the NEW token
    position_ids = torch.tensor([[input_ids.shape[1]]], device="cuda")  # first new position
    
    
    for _ in range(MAX_NEW - 1):

        if temperature > 0:
            # sample from the distribution
            probs = torch.softmax(out.logits[:, -1, :] / temperature, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
        else:
            next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_id.item())
        input_ids = torch.cat([input_ids, next_id], dim=1)
        # the "+attention_mask=None" + position_ids is what fixes it
        out = model(input_ids=next_id, position_ids=position_ids,
                    past_key_values=past, use_cache=True)
        past = out.past_key_values
        position_ids += 1
        if next_id.item() == tokenizer.eos_token_id:
            break

# torch.cuda.max_memory_allocated() / 1024**2
torch.cuda.synchronize(); dt = time.time() - t0


# Display memory usage
print(f"Peak memory usage: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")

print(tokenizer.decode(input_ids[0, prompt_len:]))
print(f"{len(generated)} tokens in {dt:.2f}s -> {len(generated)/dt:.2f} tok/s")
