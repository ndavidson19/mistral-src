import torch
from rigl_torch.RigL import RigLScheduler
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy
from mistral.brain import *
from mistral.train_data import *
import argparse
from dataclasses import asdict
import json 

# Define the command line arguments
parser = argparse.ArgumentParser(description='Transformer Model Training')
parser.add_argument('--dim', type=int, required=True, help='Model dimension')
parser.add_argument('--n_layers', type=int, required=True, help='Number of layers')
parser.add_argument('--head_dim', type=int, required=True, help='Dimension of heads')
parser.add_argument('--hidden_dim', type=int, required=True, help='Hidden dimension')
parser.add_argument('--n_heads', type=int, required=True, help='Number of heads')
parser.add_argument('--n_kv_heads', type=int, required=True, help='Number of key/value heads')
parser.add_argument('--norm_eps', type=float, required=True, help='Normalization epsilon')
parser.add_argument('--vocab_size', type=int, required=True, help='Vocabulary size')
parser.add_argument('--max_batch_size', type=int, default=0, help='Maximum batch size')
parser.add_argument('--rope_theta', type=float, default=None, help='Theta for rotary embeddings')
parser.add_argument('--sliding_window', type=int, default=None, help='Sliding window size')
parser.add_argument('--num_experts', type=int, default=8, help='Number of experts in MoE layer')
parser.add_argument('--num_experts_per_tok', type=int, default=2, help='Number of experts per token in MoE layer')
parser.add_argument('--rigl_sparsity', type=float, default=0.0, help='RigL sparsity level')
parser.add_argument('--rigl_update_freq', type=int, default=100, help='RigL update frequency')

# Parse the arguments
args = parser.parse_args()

# Convert the args to a dictionary
args_dict = vars(args)

# Create the MoEArgs object if MoE parameters are provided
if args_dict['num_experts'] is not None and args_dict['num_experts_per_tok'] is not None:
    moe_args = MoeArgs(num_experts=args_dict['num_experts'], num_experts_per_tok=args_dict['num_experts_per_tok'])
    args_dict['moe'] = moe_args
else:
    args_dict['moe'] = None

# Remove MoE parameters from the dictionary since they are now part of the MoEArgs object
del args_dict['num_experts']
del args_dict['num_experts_per_tok']

# Convert the remaining dictionary to a ModelArgs object
model_args = ModelArgs(**args_dict)

# Initialize your transformer model with the command line arguments
model = Transformer(args=model_args)

# Load Mistral's pretrained weights

mistral_weights = torch.load("mistral/modelfiles/Mistral-7B-base.pth")
model.load_state_dict(mistral_weights, strict=False)
tokenizer_model_path = 'mistral/modelfiles/tokenizer.model'


# Create the DataLoader
dataset = ProofPileDataset(split='train', tokenizer_model_path=tokenizer_model_path)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)


# Define your optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Calculate T_end
epochs = 100
total_iterations = len(dataloader) * epochs
T_end = int(0.75 * total_iterations)

# Create the RigLScheduler object
pruner = RigLScheduler(model,
                       optimizer,
                       dense_allocation=0.1,
                       sparsity_distribution='uniform',
                       T_end=T_end,
                       delta=100,
                       alpha=0.3,
                       grad_accumulation_n=1,
                       static_topo=False,
                       ignore_linear_layers=False,
                       state_dict=None)

# Training loop
for epoch in range(epochs):
    for data, labels in dataloader:
        # Forward pass, calculate loss, etc.
        optimizer.zero_grad()
        outputs = model(data)
        loss = cross_entropy(outputs, labels)  # Adjust this based on your specific task
        loss.backward()
        
        # Check if a RigL step should be performed
        if pruner():
            optimizer.step()
        
    # Checkpointing
    torch.save({
        'model': model.state_dict(),
        'pruner': pruner.state_dict(),
        'optimizer': optimizer.state_dict()
    }, f'checkpoint_{epoch}.pth')

# Print RigL scheduler information
print(pruner)

# Save model
torch.save(model.state_dict(), 'model.pth')
