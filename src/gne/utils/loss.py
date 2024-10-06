import torch
from typing import List


class Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()


class L2(Loss):
    def __init__(self, multipliers: List[float] = None):
        super().__init__()
        self.multipliers = multipliers

    def forward(self, w1, w2):
        if self.multipliers is not None:
            multipliers = self.multipliers
        else:
            multipliers = [1] * len(w1)

        if isinstance(w1, dict) and isinstance(w2, dict):
            # if w1 and w2 are dictionaries use Lagrange multipliers
            result = torch.tensor(0, dtype=torch.float64)
            keys = w1.keys()
            for k, m in zip(keys, multipliers):
                result += m * torch.linalg.norm(w1[k] - w2[k], ord=2)
            return result

        elif type(w1) is torch.Tensor and type(w2) is torch.Tensor:
            # if w1 and w2 are tensors, don't use any Lagrange multipliers
            return torch.linalg.norm(w1 - w2, ord=2)

        else:
            raise ValueError("Invalid input type")


## OLD CODE
class KullbackLeibler(Loss):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(p, q):
        logp = p.clamp(1e-20).log()
        logq = q.clamp(1e-20).log()
        return (p * logp - p * logq).sum() / len(p)


class SymKullbackLeibler(Loss):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(p, q):
        logp = p.clamp(1e-20).log()
        logq = q.clamp(1e-20).log()
        diff = q - p
        return (diff * logq - diff * logp).sum() / len(p)


class CrossEntropy(Loss):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(p, q):
        logq = q.clamp(1e-20).log()
        return -(p * logq).sum()


class SymCrossEntropy(Loss):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(p, q):
        logp = p.clamp(1e-20).log()
        logq = q.clamp(1e-20).log()
        return -(p * logq + q * logp).sum()
