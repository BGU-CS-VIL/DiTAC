import numpy as np
import difw 
from difw import Cpab 

tess_size = 7
backend = "pytorch" # ["pytorch", "numpy"]
device = "gpu" # ["cpu", "gpu"]
zero_boundary = False # [True, False]
# use_slow = False # [True, False]
grid_n_points = 100
batch_size = 1
basis = "qr" # ["svd", "sparse", "rref", "qr"]
xmin = 0
xmax = 1

T = Cpab(tess_size, backend, device, zero_boundary, basis)

grid = T.uniform_meshgrid(grid_n_points)
print('grid shape:', grid.shape)
