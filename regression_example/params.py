def get_parameters():

    params = {}
    
    # model params
    params['output_dir'] = 'outputs'    # '/vilsrv-storage/tohamy/CPAB_Activation/classification_net_with_CPAB/output/'
    params['max_it'] = 10000
    params['n_nodes'] = 100  # Number of nodes in each layer in simple_plus_fc net.
    params['n_nodes_very_simple'] = 50 
    params['batch_size'] = 98
    params['batch_size_eval'] = 500

    # Optimation params:    
    params['optimizer_type'] = 'adam'   # Choose from: [adam, SGD]
    params['lr'] = 0.005
    params['weight_decay'] = 0.0
    
    # cpab params
    params['cpab_act_type'] = 'gelu_cpab'  # Choose from: [leaky_cpab, leaky_inf_cpab, cpab_whole_range, gelu_cpab, gelu_cpab_relu (with [0, b]), gelu_cpab_gelu (with [a, 0])]
    params['tess'] = 10
    params['a'] = -3
    params['b'] = 3
    params['basis'] = 'qr'   # ['svd', 'sparse', 'rref', 'qr']
    params['lambda_smooth'] = 5   # Also called 'length_scale'
    params['lambda_var'] = 1      # Also called 'output_variance'
    params['lambda_smooth_init'] = 3   # Also called 'length_scale'
    params['lambda_var_init'] = 2       # Also called 'output_variance'    
    
    return params
