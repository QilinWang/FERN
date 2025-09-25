from nb_setup import setup_environment, reload_modules

setup_environment()
import study.FERN_util as fru
import torch

B = 3
C = 3
D = 10
H = 10

def test_rotate_factory():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.autograd.set_grad_enabled(True):
        rot_factory = fru.RotateFactory(num_reflects=3, in_dim=D, out_dim=H, channels=C, adapt_params=True)
        rot_factory.to(device)
        x = torch.randn(B, C, D, device=device, requires_grad=True)
        h = torch.randn(B, C, H, device=device)
        rot = rot_factory(h)  # build Rotation from hidden
        y = rot.rotate(x)
        x_hat = rot.rotate_back(y)
        err = (x - x_hat).pow(2).mean()
        err.backward()
        assert err.item() < 1e-6
    print("Test passed")