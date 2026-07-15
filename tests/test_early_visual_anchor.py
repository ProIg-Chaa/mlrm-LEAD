import pytest
import torch

from lead.generation_utils import _compute_early_actual_visual_anchor


def test_early_visual_anchor_selects_relevant_visual_state_and_matches_norm():
    prompt_hidden = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [0.8, 0.2], [-1.0, 0.0]]]
    )
    visual_mask = torch.tensor([[False, True, True, False]])
    query = torch.tensor([[1.0, 0.0]])
    reference = torch.tensor([[0.0, 2.0]])

    anchor, applied, similarity, ratio = _compute_early_actual_visual_anchor(
        prompt_hidden,
        visual_mask,
        query,
        reference,
        top_m=1,
        temperature=0.1,
    )

    assert applied.tolist() == [True]
    assert torch.allclose(anchor, torch.tensor([[1.940285, 0.485071]]), atol=1e-5)
    assert torch.allclose(anchor.norm(dim=-1), reference.norm(dim=-1), atol=1e-6)
    assert similarity.item() == pytest.approx(0.9701425, abs=1e-6)
    assert ratio.item() == pytest.approx(2.425356, abs=1e-5)


def test_early_visual_anchor_falls_back_without_visual_tokens():
    prompt_hidden = torch.randn(2, 4, 3)
    visual_mask = torch.zeros(2, 4, dtype=torch.bool)
    query = torch.randn(2, 3)
    reference = torch.randn(2, 3)

    anchor, applied, similarity, ratio = _compute_early_actual_visual_anchor(
        prompt_hidden,
        visual_mask,
        query,
        reference,
    )

    assert torch.equal(anchor, reference)
    assert applied.tolist() == [False, False]
    assert torch.equal(similarity, torch.zeros_like(similarity))
    assert torch.equal(ratio, torch.ones_like(ratio))


def test_early_visual_anchor_is_deterministic_and_validates_parameters():
    torch.manual_seed(42)
    prompt_hidden = torch.randn(1, 6, 5)
    visual_mask = torch.tensor([[False, True, True, True, False, False]])
    query = torch.randn(1, 5)
    reference = torch.randn(1, 5)

    first = _compute_early_actual_visual_anchor(
        prompt_hidden, visual_mask, query, reference, top_m=8, temperature=0.1
    )
    second = _compute_early_actual_visual_anchor(
        prompt_hidden, visual_mask, query, reference, top_m=8, temperature=0.1
    )
    for left, right in zip(first, second):
        assert torch.equal(left, right)

    with pytest.raises(ValueError, match="top_m"):
        _compute_early_actual_visual_anchor(
            prompt_hidden, visual_mask, query, reference, top_m=0
        )
    with pytest.raises(ValueError, match="temperature"):
        _compute_early_actual_visual_anchor(
            prompt_hidden, visual_mask, query, reference, temperature=0.0
        )
