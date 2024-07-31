import numpy as np
import matplotlib.pyplot as plt
import torch

from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator


def plot_learned_function(model, target_func_tensor, it, device, output_dir):
    dx, dy = 0.01, 0.01

    x1, x2 = np.mgrid[slice(-1, 1 + dy, dy),
                      slice(-1, 1 + dx, dx)]

    im_dim = x1.shape

    x1_tensor = torch.tensor(x1).reshape(-1,1).float()
    x2_tensor = torch.tensor(x2).reshape(-1,1).float()
    x_tensor = torch.cat((x1_tensor,x2_tensor), dim=1)

    y_pred = model(x_tensor.to(device)).reshape(im_dim).detach().cpu().numpy()
    y = target_func_tensor(x_tensor)

    y = y.reshape(im_dim).numpy()

    y = y[:-1, :-1]
    fig, (ax0, ax1) = plt.subplots(nrows=2)
    cmap = plt.colormaps['twilight_shifted']
    
    levels_y = MaxNLocator(nbins=15).tick_values(-1, 1)
    norm_y = BoundaryNorm(levels_y, ncolors=cmap.N, clip=True)
    im0 = ax0.pcolormesh(x1, x2, y, cmap=cmap, norm=norm_y)
    fig.colorbar(im0, ax=ax0)
    ax0.set_title('Actual function')
    ax0.set_xlabel('X1 (Input)', rotation="horizontal")
    ax0.set_ylabel('X2 (Input)')

    levels_y_pred = MaxNLocator(nbins=15).tick_values(-1, 1)
    norm_y_pred = BoundaryNorm(levels_y_pred, ncolors=cmap.N, clip=True)
    im1 = ax1.pcolormesh(x1, x2, y_pred, cmap=cmap, norm=norm_y_pred)
    fig.colorbar(im1, ax=ax1)
    ax1.set_xlabel('x1', rotation="horizontal")
    ax1.set_ylabel('x2')

    fig.tight_layout()
    fig.savefig(f'{output_dir}/fig1_{it}.png')
    plt.close()
