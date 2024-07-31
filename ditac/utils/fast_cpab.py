import torch

from difw.backend.pytorch.transformer import cpab_gpu


def get_fast_transform_grid(time=1.0, precision_mat=None, lambda_reg_loss=0.1, steps=2048):
    class Transformer_fast_gpu_closed_form(torch.autograd.Function):
        @staticmethod
        def forward(ctx, grid, theta, params):
            if params.xmin == 0 and params.xmax == 1:
                quant_grid = grid * (steps-1)
            else:
                quant_grid = (grid - params.xmin) * ((steps-1) * (params.xmax - params.xmin))
            quant_grid = quant_grid.round().to(torch.long)

            lin_grid = torch.linspace(params.xmin, params.xmax, steps, device=grid.device).unsqueeze(0)

            ctx.params = params
            ctx.time = time
            ctx.steps = steps
            lingrid_output = cpab_gpu.integrate_closed_form_trace(lin_grid, theta, time, params.B, params.xmin, params.xmax, params.nc)
            lingrid_t = lingrid_output[:, :, 0].contiguous()
            
            grid_t = torch.nn.functional.embedding(quant_grid, lingrid_t.T).squeeze(-1)

            ctx.save_for_backward(grid_t, grid, theta, quant_grid, lingrid_output, lin_grid)
            return grid_t
        
        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx, grad_output):  # grad [n_batch, n_points]
            output, grid, theta, quant_grid, lingrid_output, lin_grid = ctx.saved_tensors
            params = ctx.params
            time = ctx.time
            
            dphi_dtheta = cpab_gpu.derivative_closed_form_trace(
                lingrid_output, lin_grid, theta, params.B, params.xmin, params.xmax, params.nc
            )  # [1, steps, d]
            dphi_dtheta = dphi_dtheta.squeeze(0)  # [steps, d]
            dphi_dtheta = torch.nn.functional.embedding(quant_grid, dphi_dtheta)  # [n_batch, n_points, d, 1]
            dphi_dtheta = dphi_dtheta.squeeze(-1).unsqueeze(0)

            grad_theta = grad_output.mul(dphi_dtheta.permute(2, 0, 1)).sum(dim=(2)).t()

            dphi_dx = cpab_gpu.derivative_space_closed_form(
                lin_grid, theta, time, params.B, params.xmin, params.xmax, params.nc)
            dphi_dx = torch.nn.functional.embedding(quant_grid, dphi_dx.T).squeeze(-1)
            grad_x = grad_output.mul(dphi_dx)

            if precision_mat is not None:  # add gradient w.r.t. smoothness prior
                grad_theta = grad_theta + lambda_reg_loss * 2 * precision_mat.mul(theta)

            return grad_x, grad_theta, None  # [n_batch, n_points] # [n_batch, d]
        
    return Transformer_fast_gpu_closed_form().apply

# Transform data during inference, where lookup tables are prepared in advance.
def get_fast_transform_grid_eval(time=1.0, steps=2048):
    class Transformer_fast_gpu_closed_form_eval(torch.autograd.Function):
        @staticmethod
        def forward(ctx, grid, params, lingrid_t):
            if params.xmin == 0 and params.xmax == 1:
                quant_grid = grid * (steps-1)
            else:
                quant_grid = (grid - params.xmin) * ((steps-1) * (params.xmax - params.xmin))
            quant_grid = quant_grid.round().to(torch.long)
            grid_t = torch.nn.functional.embedding(quant_grid, lingrid_t.T).squeeze(-1)

            return grid_t
        
    return Transformer_fast_gpu_closed_form_eval().apply
