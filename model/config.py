# model/config.py

from dataclasses import dataclass

@dataclass
class RachanaSmall:
    vocab_size: int = 28000
    d_model: int = 384
    n_heads: int = 6
    n_layers: int = 6
    d_ff: int = 1536             # Usually 4 * d_model
    max_seq_len: int = 256       # Training time context
    dropout: float = 0.1
    attn_dropout: float = 0.1
    use_rope: bool = True
    tie_embeddings: bool = True
    use_preenorm: bool = True
    use_swiglu: bool = True
    layer_norm_eps: float = 1e-5
