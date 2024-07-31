import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import random
import numpy as np
import argparse
import torch
from torch import optim, nn
from collections import OrderedDict

from .plots import plot_learned_function
from .params import get_parameters
from .mlp import mlp_triple_2d, mlp_double_2d
from .evaluation import evaluate_test_data, compute_loss
    
    
def main(args):
    # Get args:
    seed = args.seed
    params = get_parameters()
    
    # set the seed:
    set_seed(seed)

    device = torch.device('cuda')
    
    max_it = params['max_it']
    batch_size = params['batch_size']
    batch_size_eval = params['batch_size_eval']
    output_dir = params['output_dir']
    optimizer_type = params['optimizer_type']
    lr = params['lr']
    weight_decay = params['weight_decay']

    # Get the chosen network:
    model = mlp_double_2d(params)
    model.to(device)

    # Define optimizer:
    if optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    elif optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

        # Define lr schedulers:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_it)
    
    # Loss function:
    loss_fn = nn.MSELoss()
    
    # Best mse and loss:
    best_stats = {'best_mse': 0, 'best_mse_it': 0, 'best_r2':0, 'best_loss': float('inf'), 'best_loss_it': 0}
    
    def target_function_tensor(x):
        A = torch.tensor([
            [9, -7],
            [-9, 11],
            [3, 13],
            [9, 9],
            [13, 5],
            [3,19]
        ]).float()
        B = torch.tensor([
            [0.4, 0.1, 0.15, 0.15, 0.1, 0.1]
        ]).float()

        sin_combs = torch.sin(torch.matmul(x, A.t()))  # [b, 6]
        out = torch.matmul(sin_combs, B.t())

        return out

    # Training loop:
    it = 0
    model.train()
    while it < max_it:
        
        x1 = torch.rand(batch_size, 1).float()*2 - 1 # [B, 1]
        x2 = torch.rand(batch_size, 1).float()*2 - 1 # [B, 1]
        x = torch.cat((x1, x2), dim=1)
        y = target_function_tensor(x)

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        
        y_pred = model(x)
        
        loss = compute_loss(y_pred, y, loss_fn)                    
        loss.backward()
        optimizer.step()
   
        if (it+1) % 500 == 0:
            model.eval()
            with torch.no_grad():
                stats = OrderedDict(it=it)
                stats, best_stats = evaluate_test_data(stats, best_stats, target_function_tensor, it, model, loss_fn, batch_size_eval, device)
                
                plot_learned_function(model, target_function_tensor, it, device, params['output_dir'])

            model.train()
        
        # Step for adaptive lr:
        if optimizer_type == 'SGD':
            scheduler.step()
            
        it += 1
        
    print("\n\nDone!")

            
def set_seed(seed):
    os.environ['PYTHONHASHSEED']=str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

   
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='CPAB Activation')

    parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
    
    args = parser.parse_args()
    
    main(args)
