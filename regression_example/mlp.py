import torch.nn as nn
import torch.nn.functional as F
from ditac import DiTAC


class mlp_triple_2d(nn.Module):
    
    def __init__(self, params):
        super(mlp_triple_2d, self).__init__()
        n = params['n_nodes']
        self.fc1 = nn.Linear(2, n)
        self.fc2 = nn.Linear(n, n)
        self.fc3 = nn.Linear(n, 1)

        self.act_1 = DiTAC(cpab_act_type=params['cpab_act_type'], a=params['a'], b=params['b'],
                           lambda_smooth=params['lambda_smooth'], lambda_smooth_init=params['lambda_smooth_init'],
                           lambda_var=params['lambda_var'], lambda_var_init=params['lambda_var_init'])
        self.act_2 = DiTAC(cpab_act_type=params['cpab_act_type'], a=params['a'], b=params['b'],
                           lambda_smooth=params['lambda_smooth'], lambda_smooth_init=params['lambda_smooth_init'],
                           lambda_var=params['lambda_var'], lambda_var_init=params['lambda_var_init'])
 
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act_1(x)
        x = self.fc2(x)
        x = self.act_2(x)
        y = self.fc3(x)
        return y


class mlp_double_2d(nn.Module):    
    def __init__(self, params):
        super(mlp_double_2d, self).__init__()
        n = params['n_nodes_very_simple']
        self.fc1 = nn.Linear(2, n)
        self.fc2 = nn.Linear(n, 1)
        
        self.act_1 = DiTAC(cpab_act_type=params['cpab_act_type'], a=params['a'], b=params['b'],
                           lambda_smooth=params['lambda_smooth'], lambda_smooth_init=params['lambda_smooth_init'],
                           lambda_var=params['lambda_var'], lambda_var_init=params['lambda_var_init'])
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act_1(x)
        y = self.fc2(x)
        return y    
