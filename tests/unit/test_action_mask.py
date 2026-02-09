from xppm.rl.models.masking import apply_action_mask


def test_apply_action_mask_shapes() -> None:
    import torch

    q = torch.zeros((2, 3))
    mask = torch.tensor([[1, 0, 1], [0, 1, 1]], dtype=torch.float32)
    out = apply_action_mask(q, mask)
    assert out.shape == q.shape


