import torch
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np


def R2(y, y_pred):
    # Compute R2 metric (in percentage scale 0-100):
    y_cpu = y.cpu().detach().numpy()
    y_pred_cpu = y_pred.cpu().detach().numpy()
    score = round(r2_score(y_cpu, y_pred_cpu), 4) *100
    return score


def RMSE(y, y_pred):
    y_cpu = y.cpu().detach().numpy()
    y_pred_cpu = y_pred.cpu().detach().numpy()
    score = np.sqrt(mean_absolute_error(y_cpu, y_pred_cpu))
    return score


def compute_loss(y_pred, y, loss_func):
    return loss_func(y_pred, y)
        

def evaluate_test_data(stats, best_stats, target_func_tensor, it, model, loss_fn, batch_size_eval, device):

    with torch.no_grad():
        x1, x2 = np.mgrid[slice(-1, 1 + 0.1, 0.1),
                      slice(-1, 1 + 0.1, 0.1)]

        # x1 = torch.tensor(x1).reshape(-1,1).float()
        # x2 = torch.tensor(x2).reshape(-1,1).float()
        x1 = torch.rand(batch_size_eval, 1).float()*2 - 1
        x2 = torch.rand(batch_size_eval, 1).float()*2 - 1
        x = torch.cat((x1,x2), dim=1)
        y = target_func_tensor(x).to(device)
        y_pred = model(x.to(device))
        
        loss = compute_loss(y_pred, y, loss_fn)
        r2_acc = R2(y, y_pred)
           
        # Update stats:
        stats.update({'eval_loss': loss})
        stats.update({'eval_r2_acc': r2_acc})
        
        # Update best stats:
        if loss < best_stats['best_loss']:
            best_stats.update({'best_loss': loss})
            best_stats.update({'best_loss_it': it})
            
        if r2_acc > best_stats['best_r2']:
            best_stats.update({'best_r2': r2_acc})
            best_stats.update({'best_r2_it': it})
            
        # Print stats:
        print('Test: it: {0}  Loss: {1:.3f}  r2_acc: {2:.2f}'.format(it, loss, r2_acc)) 
        
        return stats, best_stats
    