from transformers import AutoModelForCausalLM, AutoTokenizer, StopStringCriteria, StoppingCriteriaList
from huggingface_hub import login
import torch
# from optimum.gptq import GPTQQuantizers

from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig


cache_dir = '/lus/grand/projects/datascience/krishnat/model_weights/LLaMA/llama_cache/'

repo_name = "nvidia/Hymba-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(repo_name, 
                                          cache_dir = cache_dir, 
                                        #   device_map="auto",
                                          trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(repo_name,
                                             cache_dir = cache_dir,
                                            #  device_map="auto",
                                             trust_remote_code=True)

# Configure GPTQ parameters for Hymba's hybrid architecture
quant_config = QuantizeConfig(
    bits=4,                      # 4-bit quantization
    group_size=128,              # Optimal for Hymba's attention layers
    desc_act=True,               # Activation-aware quantization
    model_seqlen=2048,           # Matches Hymba's context window
    sym=True,                    # Symmetric quantization
    pack_dtype="int8",           # Efficient tensor packing
    exclude_modules=["ssm"],     # Preserve SSM layers' precision
    quant_attention_layers=3      # Quantize only full attention layers
)

# Load GPTQModel wrapper
gptq_model = GPTQModel(
    base_model=model,
    quant_config=quant_config,
    # device_map="auto"            # Automatic device placement
)

# Calibration data (Hymba-specific recommendation)
calib_data = load_dataset("nvidia/hymba-calibration", split="train")["text"][:1024]

# Execute quantization
gptq_model.quantize(
    calibration_data=calib_data,
    batch_size=8,                # Adjust based on VRAM (24GB+ recommended)
    use_autoround=True           # Enable advanced rounding (2025 feature)
)


# model = model.cuda().to(torch.bfloat16)
model = gptq_model.cuda()

def chat_with_model(messages, model, tokenizer, max_new_tokens=256):
    tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to('cuda')
    stopping_criteria = StoppingCriteriaList([StopStringCriteria(tokenizer=tokenizer, stop_strings="</s>")])
    outputs = model.generate(
        tokenized_chat, 
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.7,
        use_cache=True,
        stopping_criteria=stopping_criteria
    )
    input_length = tokenized_chat.shape[1]
    response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    return response
messages = [
    {"role": "system", "content": "You are a helpful assistant."}
]
print("Chat with the model (type 'exit' to quit):")
while True:
    print("User:")
    prompt = input()
    if prompt.lower() == "exit":
        break
    
    messages.append({"role": "user", "content": prompt})
    response = chat_with_model(messages, model, tokenizer)
    messages.append({"role": "assistant", "content": response})
    
    print(f"Model: {response}")