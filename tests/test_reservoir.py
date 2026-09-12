import numpy as np
import pytest
import torch
from mindmeld.reservoir import make_csr, to_torch, step, Projection, signals

def test_orientation():
    # 0 -> 1 (2), 1 -> 2 (-3); source state must accumulate at destination.
    w = to_torch(make_csr([0,1], [1,2], [2,-3], 3), "cpu")
    np.testing.assert_allclose(torch.mv(w, torch.tensor([5.,7.,11.])).numpy(), [0,10,-21])

def test_dynamics_and_recurrence():
    w = to_torch(make_csr([0,1], [1,0], [.9,-.5], 2), "cpu")
    x = torch.tensor([.1,-.2]); drive = torch.tensor([.4,.3])
    expected = .85*x + .15*torch.tanh(torch.tensor([.1,.09])+drive)
    torch.testing.assert_close(step(w,x,drive),expected)
    assert not torch.allclose(step(w,x,drive),.85*x+.15*torch.tanh(drive))
    for _ in range(100): x=step(w,x,drive)
    assert torch.isfinite(x).all() and x.abs().max() <= 1

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cpu_gpu_and_seed():
    scipy_w = make_csr([0,1,2],[1,2,0],[.7,-.6,.5],3)
    inputs = signals(100)
    results=[]
    for device in ["cpu","cuda","cuda"]:
        w=to_torch(scipy_w,device); p=Projection(3,device); x=torch.zeros(3,device=device)
        for u in torch.tensor(inputs,device=device): x=step(w,x,p(u))
        results.append(x.cpu())
    torch.testing.assert_close(results[0],results[1],atol=2e-6,rtol=2e-5)
    assert torch.equal(results[1],results[2])
