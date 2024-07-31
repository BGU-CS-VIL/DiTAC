import torch
import copy
import difw

from difw.backend.pytorch.transformer import cpab_gpu

from .utils import get_fast_transform_grid, get_fast_transform_grid_eval
from .utils import get_precision_mat, norm_data, phi
from .utils import cpab_T


class DiTAC(torch.nn.Module):

    def __init__(self, 
                 cpab_act_type='gelu_cpab',
                 tess=10,
                 a=-0.5,
                 b=2,
                 basis='qr',
                 lambda_smooth=5,
                 lambda_smooth_init=5,
                 lambda_var=2,
                 lambda_var_init=2,
                 lambda_reg_loss=0.1):
        
        super(DiTAC, self).__init__()

        self.cpab_act_type = cpab_act_type
        self.tess = tess
        self.a = a
        self.b = b
        self.xmin = 0
        self.xmax = 1
        self.zero_bndry = True
        self.basis = basis
        self.backend = 'pytorch'
        self.cpab_device = 'gpu'
        self.d = self.tess - 1
        self.lambda_smooth = lambda_smooth
        self.lambda_var = lambda_var
        self.lambda_smooth_init = lambda_smooth_init
        self.lambda_var_init = lambda_var_init
        self.lambda_reg_loss = lambda_reg_loss

        self.steps = 2**11  # number of intervals in the lookup tables in fast_cpab
        self.lingrid_t = None
        
        t = difw.Cpab(tess_size=copy.deepcopy(self.tess), 
                            backend=copy.deepcopy(self.backend), 
                            device=copy.deepcopy(self.cpab_device), 
                            zero_boundary=copy.deepcopy(self.zero_bndry), 
                            basis=copy.deepcopy(self.basis))
    
        # Initialize theta: (sample with prior)
        thetas = t.sample_transformation_with_prior(1, length_scale=self.lambda_smooth_init, output_variance=self.lambda_var_init).to(torch.device('cuda'))
        self.T = cpab_T(self.tess, self.zero_bndry, self.basis, self.cpab_device)
        
        # Set all thetas as network parameters:   
        self.thetas = torch.nn.Parameter(data=thetas, requires_grad=True)  # [1, theta_dim]

        precision_mat = get_precision_mat(self.T.params.D, self.T.params.B, self.lambda_smooth, self.lambda_var)
        self.precision_mat = torch.nn.Parameter(precision_mat, requires_grad=False)
        self.fast_transform_grid = get_fast_transform_grid(precision_mat=self.precision_mat, lambda_reg_loss=self.lambda_reg_loss, steps=self.steps)
        self.fast_transform_grid_eval = get_fast_transform_grid_eval(steps=self.steps)

        # Define cpab_act function:
        self.act = None
        if self.cpab_act_type == 'gelu_cpab':
            self.act = self.gelu_cpab
        elif self.cpab_act_type == 'leaky_cpab':
            raise NotImplementedError()
        else:
            raise ValueError()

    def gelu_cpab(self, data):
        shape = data.shape  # [bs, d]
        data = data.reshape(-1)  # [bs * d]

        data_t = self.transform_data(data)  # Transform data with CPAB
        data_t = data_t * phi(data)  # GELU-like transform, T_theta(x)*phi(x) instead of x*phi(x)
        data_t = data_t.reshape(shape)

        return data_t
    
    def transform_data(self, data):
        # Normalize to Omega only the data that is originaly between [a, b]
        in_a_b_range_ids = torch.logical_and(data >= self.a, data <= self.b)
        data_in_a_b_range = torch.where(in_a_b_range_ids, data, 0)
        data_norm = norm_data(data_in_a_b_range, self.a, self.b, self.xmin, self.xmax)  # normilize in proportion of [a,b] because we only take data thats originally from this range

        # Apply CPAB:
        if not self.training:
            data_norm_t = self.fast_transform_grid_eval(data_norm, self.T.params, self.lingrid_t)  # [1, bs * d]
        else:
            data_norm_t = self.fast_transform_grid(data_norm, self.thetas, self.T.params)  # [1, bs * d]
        data_norm_t = data_norm_t.reshape(-1)

        # Scale the transformed data back to iriginal range
        data_t = norm_data(data_norm_t, self.xmin, self.xmax, self.a, self.b)
        cpab_out = torch.where(in_a_b_range_ids, data_t, data)

        return cpab_out
    
    def prep_for_eval(self):
        # Prepare lookup tables for fast inference:
        with torch.no_grad():
            params = self.T.params
            lin_grid = torch.linspace(params.xmin, params.xmax, self.steps, device=self.thetas.device).unsqueeze(0)
            lingrid_output = cpab_gpu.integrate_closed_form_trace(lin_grid, self.thetas, 1.0, params.B, params.xmin, params.xmax, params.nc)
            self.lingrid_t = lingrid_output[:, :, 0].contiguous()
    
    def eval(self):
        self.prep_for_eval()
        return self.train(False)

    def train(self, mode=True):
        super().train(mode)
        if not mode:
            self.prep_for_eval()
        return self

    def forward(self, x):
        out = self.act(x)
        return out
