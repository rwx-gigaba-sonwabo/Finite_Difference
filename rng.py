
from __future__ import annotations

from dataclasses import dataclass

import torch


def norm_icdf(u: torch.Tensor) -> torch.Tensor:
    """
    RiskFlow equivalent:
        return sqrt(2) * erfinv(2u - 1)

    This matches RiskFlow's utils.norm_icdf.
    """
    return 1.4142135623730951 * torch.erfinv(2.0 * u - 1.0)


@dataclass
class SobolNormalRng:
    """
    Scrambled Sobol -> U(0,1) -> Normal(0,1) via inverse CDF.
    Mirrors RiskFlow's pattern (SobolEngine + (0.5 + (1-eps)*(x-0.5)) + norm_icdf).
    """
    seed: int
    fast_forward: int = 0
    device: torch.device | str = "cpu"
    dtype: torch.dtype = torch.float64

    def draw_normals(self, dimension: int, n: int) -> torch.Tensor:
        """
        Returns tensor of shape (dimension, n) of N(0,1).
        """
        engine = torch.quasirandom.SobolEngine(
            dimension=dimension,
            scramble=True,
            seed=self.seed,
        )
        if self.fast_forward > 0:
            engine.fast_forward(self.fast_forward)

        sobol = engine.draw(n, dtype=self.dtype).to(self.device)
        # Avoid exact 0/1 boundaries (RiskFlow does a symmetric epsilon-shift around 0.5)
        eps = torch.finfo(sobol.dtype).eps
        u = (0.5 + (1.0 - eps) * (sobol - 0.5)).to(self.device)

        z = norm_icdf(u)  # (n, dimension)
        return z.transpose(0, 1).contiguous()  # (dimension, n)
