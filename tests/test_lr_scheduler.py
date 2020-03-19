import pytest
import torch
from pytest import approx
from torch.nn import Parameter
from torch.optim import AdamW  # type: ignore
from torch.optim.lr_scheduler import CyclicLR

from gdmtl.utils import get_linear_schedule_with_warmup_and_minimum


def _run_scheduler(steps: int) -> float:
    params = Parameter(torch.tensor([0], dtype=torch.float))  # type: ignore
    optimizer = AdamW([params], lr=1e-5)
    scheduler = get_linear_schedule_with_warmup_and_minimum(
        optimizer, 100, 1000, min_ratio=0.1, num_work_steps=200
    )
    for _ in range(steps):
        optimizer.step()
        scheduler.step()

    return scheduler.get_last_lr()[0]  # type:ignore


def test_lr_scheduler() -> None:
    # 1 step
    assert _run_scheduler(1) == approx(1e-5 * (1 / 100))

    # 50 step
    assert _run_scheduler(50) == approx(1e-5 * (50 / 100))

    # 100 steps, maximum
    assert _run_scheduler(100) == approx(1e-5)

    # 200 steps, stabilize
    assert _run_scheduler(200) == approx(1e-5)

    # 350 steps, decay
    assert _run_scheduler(350) == approx(1e-5 * (650 / 700))

    # 930 steps, minimum
    assert _run_scheduler(930) == approx(1e-6)

    # 980 steps, minimum
    assert _run_scheduler(980) == approx(1e-6)


def test_zero_work_step() -> None:
    params = Parameter(torch.tensor([0], dtype=torch.float))  # type: ignore
    optimizer = AdamW([params], lr=1e-5)
    scheduler = get_linear_schedule_with_warmup_and_minimum(
        optimizer, 100, 1000, min_ratio=0.1
    )
    for _ in range(101):
        optimizer.step()
        scheduler.step()

    assert scheduler.get_last_lr()[0] == approx(1e-5 * (899 / 900))


def _run_cyclic_scheduler(steps: int) -> float:
    params = Parameter(torch.tensor([0], dtype=torch.float))  # type: ignore
    optimizer = AdamW([params], lr=1e-4)
    scheduler = CyclicLR(
        optimizer, base_lr=1e-6, max_lr=1e-4, step_size_up=100, cycle_momentum=False
    )
    for _ in range(steps):
        optimizer.step()
        scheduler.step()  # type: ignore

    return scheduler.get_last_lr()[0]  # type:ignore


@pytest.mark.parametrize(  # type:ignore
    "steps, lr",
    [(1, 1e-6 + (1e-4 - 1e-6) / 100), (100, 1e-4), (200, 1e-6)],
)
def test_cyclic_scheduler(steps: int, lr: float) -> None:
    # 100 step
    assert _run_cyclic_scheduler(steps) == approx(lr)
