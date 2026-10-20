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

MAX_NEW = 30
generated = []

torch.cuda.synchronize(); t0 = time.time()
with torch.inference_mode():
    for _ in range(MAX_NEW):
        out = model(input_ids=input_ids, attention_mask=attn)   # forward on FULL prefix
        next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)   # greedy = argmax
        generated.append(next_id.item())
        input_ids = torch.cat([input_ids, next_id], dim=1)
        attn = torch.cat([attn, torch.ones_like(next_id)], dim=1)
        if next_id.item() == tokenizer.eos_token_id:
            break
torch.cuda.synchronize(); dt = time.time() - t0

print(tokenizer.decode(input_ids[0, prompt_len:]))
print(f"{len(generated)} tokens in {dt:.2f}s -> {len(generated)/dt:.2f} tok/s")