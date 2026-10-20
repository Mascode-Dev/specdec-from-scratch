import torch
import transformers

model = transformers.GPT2LMHeadModel.from_pretrained("gpt2", torch_dtype=torch.float16).to("cuda")
tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2")
model.eval()

prompt = "Hello, how are you? I'm fine, thanks for"
MAX_NEW = 50
enc = tokenizer(prompt, return_tensors="pt").to("cuda")
input_ids, attn = enc["input_ids"], enc["attention_mask"]
prompt_len = input_ids.shape[1]

# 1) Hand-rolled greedy with KV cache
with torch.inference_mode():
    out = model(input_ids=input_ids, attention_mask=attn, use_cache=True)
    past = out.past_key_values
    hand = []
    pos = torch.tensor([[input_ids.shape[1]]], device="cuda")
    for _ in range(MAX_NEW):
        nid = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        hand.append(nid.item())
        out = model(input_ids=nid, position_ids=pos, past_key_values=past, use_cache=True)
        past, pos = out.past_key_values, pos + 1
hand_ids = torch.tensor(hand, device="cuda")

# 2) HF generate, greedy
gen = model.generate(
    **enc,
    max_new_tokens=MAX_NEW,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id
)

gen_ids = gen[0, prompt_len:]

# 3) Compare
print("match:", torch.equal(hand_ids, gen_ids))
if not torch.equal(hand_ids, gen_ids):
    for i, (a, b) in enumerate(zip(hand_ids.tolist(), gen_ids.tolist())):
        if a != b:
            print(f"diff@{i}: hand={a}  gen={b}"); break
    print(f"len hand={len(hand_ids)}  gen={len(gen_ids)}")