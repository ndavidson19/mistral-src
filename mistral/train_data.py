from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from mistral.tokenizer import Tokenizer

class ProofPileDataset(Dataset):
    def __init__(self, split, tokenizer_model_path):
        #self.dataset = load_dataset("EleutherAI/proof-pile-2", 'default', split=split)
        self.dataset = load_dataset("roneneldan/TinyStories", split=split)
        self.tokenizer = Tokenizer(model_path=tokenizer_model_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx]["text"]
        input_ids = self.tokenizer.encode(text, bos=True)
        # Convert input_ids to a PyTorch tensor
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        # Create an attention mask (1 for real tokens, 0 for padding)
        attention_mask = torch.ones_like(input_ids)
        return input_ids, attention_mask

def collate_fn(batch):
    # Find the longest sequence in the batch
    max_len = max(len(item[0]) for item in batch)
    # Pad all sequences to the max length
    padded_input_ids = torch.full((len(batch), max_len), fill_value=tokenizer.pad_id, dtype=torch.long)
    padded_attention_mask = torch.zeros_like(padded_input_ids)
    for i, (input_ids, attention_mask) in enumerate(batch):
        padded_input_ids[i, :len(input_ids)] = input_ids
        padded_attention_mask[i, :len(attention_mask)] = attention_mask
    return padded_input_ids, padded_attention_mask

