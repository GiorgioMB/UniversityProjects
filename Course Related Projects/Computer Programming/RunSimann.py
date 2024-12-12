import numpy as np
from ClassFinal import Grid
from SimAnnFinal import simann
from matplotlib import pyplot as plt

n = 100
grid = Grid(n, "3232295")
steps = 10 * n
best_grid, best_c = simann(grid, 0.01, 50, 50, n, 3232295)
