import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import time
from model.config import RachanaSmall
from model.rachana_small import RachanaSmallModel

def run_extended_dry_tests():
    print("🚀 Initializing config & model...")
    config = RachanaSmall()
    model = RachanaSmallModel(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print("✅ Model initialized. Param count:", sum(p.numel() for p in model.parameters()) // 1_000, "K")

    # Create dummy input
    batch_size = 2
    seq_len = config.max_seq_len
    vocab_size = config.vocab_size
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)

    print(f"🧪 Forward pass test (batch={batch_size}, seq_len={seq_len})")
    start_time = time.time()
    logits = model(input_ids)
    end_time = time.time()
    print("✅ Logits shape:", logits.shape)
    print("⏱️ Time per forward pass:", round(end_time - start_time, 4), "seconds")

    # Force non-zero loss (by shifting tokens)
    labels = input_ids.roll(shifts=1, dims=1)
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits.view(-1, vocab_size), labels.view(-1))
    print("🎯 Loss (misaligned labels):", round(loss.item(), 4))

    # Backward pass
    loss.backward()
    print("✅ Backward pass complete.")

    # Gradient norms for all attention blocks
    for i, block in enumerate(model.blocks):
        try:
            norm_val = block.attn.norm.weight.grad.norm().item()
            print(f"🔧 Grad norm (block {i}): {round(norm_val, 6)}")
        except:
            print(f"⚠️ Grad norm (block {i}): not available")

    # Param breakdown
    total_params = 0
    print("\n🧮 Param breakdown:")
    for name, param in model.named_parameters():
        count = param.numel()
        total_params += count
        print(f"{name:<60} {count//1_000}K")
    print(f"✅ Total parameters: {total_params // 1_000}K")

    # Save checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/test_pass2_model.pt")
    print("💾 Model checkpoint saved to 'checkpoints/test_pass2_model.pt'")

if __name__ == "__main__":
    run_extended_dry_tests()
