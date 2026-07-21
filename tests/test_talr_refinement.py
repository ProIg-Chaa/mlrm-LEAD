import pytest

from lead.generation_utils import _talr_refinement_eligible


@pytest.mark.parametrize("window", [8, 16, 32])
@pytest.mark.parametrize("soft_cap", [1, 2])
def test_refinement_is_allowed_inside_window_below_cap(window, soft_cap):
    assert _talr_refinement_eligible(
        step=5,
        transition_step=1,
        refinement_count=0,
        window=window,
        soft_cap=soft_cap,
        entropy_proposed=True,
    )


def test_refinement_rejects_transition_step_and_expired_window():
    assert not _talr_refinement_eligible(
        step=1,
        transition_step=1,
        refinement_count=0,
        window=8,
        soft_cap=1,
        entropy_proposed=True,
    )
    assert not _talr_refinement_eligible(
        step=10,
        transition_step=1,
        refinement_count=0,
        window=8,
        soft_cap=1,
        entropy_proposed=True,
    )


def test_refinement_rejects_cap_lock_and_missing_proposal():
    common = {
        "step": 4,
        "transition_step": 1,
        "window": 8,
        "soft_cap": 1,
    }
    assert not _talr_refinement_eligible(
        **common, refinement_count=1, entropy_proposed=True
    )
    assert not _talr_refinement_eligible(
        **common,
        refinement_count=0,
        entropy_proposed=True,
        locked_normal=True,
    )
    assert not _talr_refinement_eligible(
        **common, refinement_count=0, entropy_proposed=False
    )
