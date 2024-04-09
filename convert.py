from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load the model from the Hugging Face model hub
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", quantization_config=bnb_config)

# Save the model in the standard PyTorch format
torch.save(model.state_dict(), "Mistral-7B-Instruct-v0.2_converted.pth")
