import random
import numpy as np
import matplotlib.pyplot as plt
def accept_with_prob(delta_cost, beta, betamax):
    if beta > betamax:
        return False
    if delta_cost <= 0:
        return True
    
    prob = np.exp(-beta * delta_cost)
    return np.random.random() < prob #True with the correct probaility, otherwise false 

# meta-heuristic method for solving an optimization problem.
# One needs to define:
# - probl.init_config()
# - probl.cost()
# - probl.display()
# - move = probl.propose_move()
# - probl.accept(move)
# - delta_c = compute_delta_cost(move)
def simann(probl, beta0=0.1, beta1=10., anneal_steps=10,
           mcmc_steps=10, seed=None, wait = 10, wait_2 = 30):
    """
    Inputs:
        -probl: the class of the problem you want to solve
        -beta0: the maximum temperature - the minimum beta value
        -beta1: the minimum temperature - the maximum beta value
        -anneal_steps: number of steps that are between beta0 and beta1
        -mcmc_steps: Markov Chain Monte Carlo steps for each beta value
        -seed: random seed (for debugging purposes)
        -wait: number of times the algorithm is allowed to accept 0 moves before halting
        -wait_2: number of times the algorithm is allowed to not improve the best cost before halting
    A better implementation of the Simulated Annealing, as now
    if no moves are accepted for 'wait' times in a row, the code
    stops early. Also, if for 'wait_2' times in a row, a better cost
    isn't found, it stops. It also plots the frequency moves are accepted
    over the beta value and changed the way the betas are 
    calculated, putting together two linear spaces, with a third of the
    anneal_steps dedicated to the space that goes from beta0 to log_10(beta1)
    and the other two thirds from log_10(beta1) to beta1
    """
    #print(f"Running version 7 of SimAnn, class of problem: {probl.name()}")
    
    if seed is not None:
        np.random.seed(seed)
    best_c = probl.cost()
    cx = best_c
    #print(f"initial cost is c={best_c}")
    best_probl = probl.copy()
    third_anneal_steps = anneal_steps // 3
    betas = np.concatenate([
        np.linspace(beta0, np.log10(beta1), third_anneal_steps),
        np.linspace(np.log10(beta1), beta1, anneal_steps - third_anneal_steps)
    ])
    betas = np.append(betas, np.inf)
    jumps = np.arange(1, anneal_steps + 2)
    #print(betas)
    patience, second_patience = wait, 0
    frequency = []
    idx = 0
    for beta in betas:
        idx += 1
        probl.cost_cache = {}
        accepted_moves = 0
        second_patience += 1
        for t in range(mcmc_steps):
            move = probl.propose_move()
            delta_c = probl.compute_delta_cost(move)
            current_x, current_y = probl.position_x, probl.position_y
            dx, dy = move
            random_idx = np.random.randint(0, idx, 2)
            multiplier_x, multiplier_y = jumps[random_idx[0]], jumps[random_idx[1]]
            #print(random_multiplier)
            newdx, newdy = dx * multiplier_x, dy * multiplier_y
            move = (newdx, newdy)


            if accept_with_prob(delta_c, beta, beta1):
                accepted_moves += 1
                probl.accept_move(move)
                cx = probl.cost()
                if best_c > cx:
                    second_patience = 0
                    best_c = cx
                    best_probl = probl.copy()
                    #(f"Sanity check: {best_probl.f_values[best_probl.position_x, best_probl.position_y] == best_c}")

           
        #print(f"Sanity check: {best_probl.f_values[best_probl.position_x, best_probl.position_y] == best_c}")
        if second_patience == wait_2 or patience == 0:
            break
        patience = wait if accepted_moves > 0 else patience - 1
        #best_probl.display_beta(beta)
        #print(f"beta={beta} accept_freq={accepted_moves/mcmc_steps} c={cx} best_c={best_c}")
        frequency.append(accepted_moves/mcmc_steps)
        
    ## means they've been removed for efficiency tests
    ##best_probl.display()
    print(f"best cost = {best_c}")
    plot_betas = betas[:len(frequency)]
    plt.plot(plot_betas,frequency,marker = ".")
    
    return best_probl, best_c