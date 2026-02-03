import torch


def test_within_mask_gates_out_values():
    """
    If within_mask=0, within-season values must have zero influence on z.

    This is our practical guardrail against "fake zeros" or arbitrary imputation
    leaking into the embedding when game-log windows are missing.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from models.player_encoder import PlayerEncoder

    torch.manual_seed(0)
    enc = PlayerEncoder(tier1_dim=3, tier2_dim=2, career_dim=3, within_dim=2, latent_dim=8, use_vae=False)
    enc.eval()

    tier1 = torch.tensor([[1.0, 2.0, 3.0]])
    tier2 = torch.tensor([[9.0, 9.0]])
    career = torch.tensor([[2.0, 0.1, -0.2]])

    # within differs wildly, but within_mask=0 should null it out.
    within_a = torch.tensor([[0.0, 0.0]])
    within_b = torch.tensor([[1000.0, 999.0]])

    tier2_mask = torch.tensor([[1.0]])
    within_mask = torch.tensor([[0.0]])

    z_a = enc.encode(tier1, tier2, career, within_a, tier2_mask, within_mask)
    z_b = enc.encode(tier1, tier2, career, within_b, tier2_mask, within_mask)

    assert torch.allclose(z_a, z_b, atol=0.0, rtol=0.0)


def test_tier2_mask_gates_out_values():
    """If tier2_mask=0, spatial values must have zero influence on z."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from models.player_encoder import PlayerEncoder

    torch.manual_seed(0)
    enc = PlayerEncoder(tier1_dim=3, tier2_dim=2, career_dim=3, within_dim=2, latent_dim=8, use_vae=False)
    enc.eval()

    tier1 = torch.tensor([[1.0, 2.0, 3.0]])
    career = torch.tensor([[2.0, 0.1, -0.2]])
    within = torch.tensor([[0.0, 0.0]])
    within_mask = torch.tensor([[1.0]])

    tier2_a = torch.tensor([[0.0, 0.0]])
    tier2_b = torch.tensor([[999.0, -999.0]])
    tier2_mask = torch.tensor([[0.0]])

    z_a = enc.encode(tier1, tier2_a, career, within, tier2_mask, within_mask)
    z_b = enc.encode(tier1, tier2_b, career, within, tier2_mask, within_mask)

    assert torch.allclose(z_a, z_b, atol=0.0, rtol=0.0)


if __name__ == "__main__":
    test_within_mask_gates_out_values()
    test_tier2_mask_gates_out_values()
    print("OK")
