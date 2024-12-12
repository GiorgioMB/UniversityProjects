#%%
import numpy as np
from Class7 import Grid
from SimAnn6 import simann
from matplotlib import pyplot as plt

##Before running it, uncomment probl.display_beta(beta) in SimAnn6.py, line 93
n = 100
grid = Grid(n, "3232295")
steps = 10 * n
best_grid, best_c = simann(grid, 0.01, 50, 50, n, 3232295)



# %%
