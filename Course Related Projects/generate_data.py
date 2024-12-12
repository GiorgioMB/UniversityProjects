#%%
import numpy as np  
import matplotlib.pyplot as plt      

def generate_data(n, seed, np_seed=None):
    if np_seed is not None:
        np.random.seed(np_seed)
    if type(seed) != str:
        raise TypeError("The seed should be the string representing your ID")
    if type(n) != int:
        raise TypeError("The dimension of the problem should be an integer value")
    if n%2 != 0:
        raise ValueError("The dimension of the problem should be even!!")
    
    aggregate_counter = 0
    for char in seed:
        aggregate_counter += int(char)
    
    aggregate_counter = 10*aggregate_counter
    marginal_diff = int(aggregate_counter)%n 
    bsize = 12 + int(np.floor(n**(1/2)))
    if (bsize%2)!=0:
        bsize=bsize-1
    
    if (marginal_diff<bsize+4):
        marginal_diff = bsize+4  
    if (marginal_diff > (n-bsize-4)):
        marginal_diff = n-bsize-4
    if ((marginal_diff > n/2 - bsize-5) and (marginal_diff < n/2+bsize+5)):
        marginal_diff = int(n/2-bsize-5)
    
    c_marginal_diff = n-marginal_diff
    
    b1 = -100 + np.random.randn(bsize, bsize)
    b2 = -50 + np.random.randn(bsize, bsize)
    

    f_values = np.zeros((n,n), dtype=np.float32)
    f_values = 1+ 0.05*np.random.randn(n,n)
    f_values[marginal_diff-bsize//2:marginal_diff+bsize//2, c_marginal_diff-bsize//2:c_marginal_diff+bsize//2] = b1
    f_values[c_marginal_diff-bsize//2:c_marginal_diff+bsize//2, marginal_diff-bsize//2:marginal_diff+bsize//2] = b2
    return f_values
    
# %%
