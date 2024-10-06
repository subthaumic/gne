import torch
import pytest

from gne.utils.loss import L2


def test_l2_loss_without_multipliers():
    l2_loss = L2()
    w1 = torch.tensor([1, 1, 1], dtype=float)
    w2 = torch.tensor([0, 0, 0], dtype=float)
    expected_result = torch.linalg.norm(w1 - w2, ord=2)
    assert l2_loss(w1, w2) == expected_result


def test_l2_loss_with_multipliers():
    l2_loss = L2([1, 0.5])
    w1 = {0: torch.tensor([1, 1, 1], dtype=float), 1: torch.tensor([2, 2], dtype=float)}
    w2 = {0: torch.tensor([0, 0, 0], dtype=float), 1: torch.tensor([1, 1], dtype=float)}
    expected_result = 1 * torch.linalg.norm(
        w1[0] - w2[0], ord=2
    ) + 0.5 * torch.linalg.norm(w1[1] - w2[1], ord=2)
    assert l2_loss(w1, w2) == expected_result


def test_l2_loss_with_multipliers_single_tensor():
    l2_loss = L2([2, 0.5])
    w1 = torch.tensor([1, 1, 1], dtype=float)
    w2 = torch.tensor([0, 0, 0], dtype=float)
    expected_result = torch.linalg.norm(w1 - w2, ord=2)
    assert l2_loss(w1, w2) == expected_result


def test_l2_loss_with_multipliers_invalid_input():
    l2_loss = L2()
    w1 = {0: torch.tensor([1, 1, 1], dtype=float), 1: torch.tensor([2, 2], dtype=float)}
    w2 = torch.tensor([0, 0, 0], dtype=float)
    with pytest.raises(ValueError):
        l2_loss(w1, w2)
