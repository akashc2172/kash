"""
Temporal Attention Encoder (Phase 2 Design)
===========================================

Per-season MLP → exposure-aware attention pooling → z_career

Key Components:
1. Per-Season MLP: Shared weights, inputs include transfer_flag
2. Padding: T_max=6, mask_t for valid seasons
3. Attention: e_t = v·tanh(W_h·h_t + W_p·log(poss_t+1))
4. Exposure-aware: log(poss) in attention score
5. Regularization: Attention dropout + entropy bonus
6. Logging: attention_weights for interpretability

See docs/temporal_attention_design.md for full specification.
"""

# Placeholder for Phase 2 implementation
# Full design documented in /Users/akashc/.windsurf/plans/temporal_attention_design.md
