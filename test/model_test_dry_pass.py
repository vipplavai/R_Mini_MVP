# test/test_model_dry_pass.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from model.config import RachanaSmall
from model.rachana_small import MiniMistralModel



import os
os.makedirs("checkpoints", exist_ok=True)

def run_dry_pass():
    print("🚀 Initializing config & model...")
    config = RachanaSmall()
    model = MiniMistralModel(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print("✅ Model initialized. Param count:", sum(p.numel() for p in model.parameters()) // 1_000, "K")

    # Create dummy input
    batch_size = 2
    seq_len = config.max_seq_len
    vocab_size = config.vocab_size

    print(f"🧪 Running dry forward/backward pass with batch_size={batch_size}, seq_len={seq_len}")
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)

    # Forward pass
    logits = model(input_ids)
    print("✅ Forward pass complete. Logits shape:", logits.shape)

    # Dummy labels (same as input for causal LM)
    labels = input_ids.clone()

    # Compute loss
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits.view(-1, vocab_size), labels.view(-1))
    print("🎯 Loss:", loss.item())

    # Backward pass
    loss.backward()
    grad_norm = model.blocks[0].attn.norm.weight.grad.norm().item()
    print("✅ Backward pass complete. Sample grad norm:", grad_norm)

    # Save checkpoint (optional)
    torch.save(model.state_dict(), "checkpoints/dry_pass_model.pt")
    print("💾 Model checkpoint saved to 'checkpoints/dry_pass_model.pt'")

if __name__ == "__main__":
    run_dry_pass()
