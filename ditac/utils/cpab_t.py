import torch
from . import Parameters, Tessellation


def to(x, dtype=torch.float32, device=None):
    if type(device) == str:
        device = torch.device("cuda") if device == "gpu" else torch.device("cpu")
    if torch.is_tensor(x):
        return x.detach().clone().type(dtype).to(device)
    return torch.tensor(x, dtype=dtype, device=device)

class cpab_T():
    # A class that stores 
    #   T.params.D, T.params.d, T.params.B, params.xmin, params.xmax, params.nc
    # to be used during training
    # This is a hack to avoid saving cpab object as class attribute in CpabAct (as it cause error- the module can;t be pickled).
    def __init__(self, tess_size, zero_boundary, basis, device):
        self.params = Parameters()
        self.params.nc = tess_size
        self.params.xmin = 0
        self.params.xmax = 1
        self.params.zero_boundary = zero_boundary
        self.params.basis = basis
        self.device = device
        
        # Initialize tesselation
        self.tess = Tessellation(
            self.params.nc, self.params.xmin, self.params.xmax, self.params.zero_boundary, basis=self.params.basis,
        )

        # Extract parameters from tesselation
        self.params.B = self.tess.B
        self.params.D, self.params.d = self.tess.D, self.tess.d
        
        self.params.B = to(self.params.B, device=self.device)
        self.params.B = self.params.B.contiguous()
