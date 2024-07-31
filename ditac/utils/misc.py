import torch
import math
import numpy as np

def norm_data(data, origin_min, origin_max, target_min, target_max):
    delta_target = target_max - target_min
    delta_origin = origin_max - origin_min

    add_part = - (origin_min - (target_min * delta_origin / delta_target))
    mul_part = delta_target / delta_origin

    data = data + add_part
    data = data * mul_part

    return data


def phi(x):
    return 0.5 * (1 + torch.erf(x / math.sqrt(2)))


def get_precision_mat(D, B, lambda_smooth, lambda_var):
    with torch.no_grad():
        if isinstance(B, np.ndarray):
            B = torch.from_numpy(B).float()
        
        B = B.to('cpu')

        # Distance between centers
        centers = torch.linspace(-1., 1., D)  # from 0 to 1 with nC steps

        # calculate the distance
        dists = torch_dist_mat(centers)  # DxD

        # Covariance in PA space:
        cov_avees = torch.exp(-(dists / lambda_smooth))
        cov_avees *= (cov_avees * (lambda_var * D) ** 2)

        B_T = torch.transpose(B, 0, 1)
        cov_cpa = torch.matmul(B_T, torch.matmul(cov_avees, B))
        eps = (torch.eye(cov_cpa.shape[0])*torch.finfo(torch.float32).eps)
        precision_cpa = torch.inverse(cov_cpa + eps)

        return precision_cpa


def torch_dist_mat(centers):
    '''
    Produces an NxN  dist matrix D,  from vector (centers) of size N
    Diagnoal = 0, each entry j, represent the distance from the diagonal
    dictated by the centers vector input
    '''
    times = centers.shape  # Torch.Tensor([n], shape=(1,), dtype=int32)

    # centers_grid tile of shape N,N, each row = centers
    centers_grid = centers.repeat(times[0],1)
    dist_matrix = torch.abs(centers_grid - torch.transpose(centers_grid, 0, 1))
    return dist_matrix
