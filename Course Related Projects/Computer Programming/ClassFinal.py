#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from copy import deepcopy
from generate_data import generate_data

class Grid:
    """
    Input:
        - n: dimension of the problem
        - individual_id: your student ID
        Notes: 
            -- n must be an integer, even and >68
            -- individual_id must be a string
        Failing to match these conditions will result in the class throwing an error
    """
    def __init__(self, n, individual_id):
        if n < 70 or n % 2 != 0:
            raise ValueError(f"{n} is invalid. Input number must be even and >68")
        if not type(n) == int:
            raise TypeError(f"{n} is invalid. Input number must be an integer")
        if not type(individual_id) == str:
            raise TypeError(f"{individual_id} is invalid. Input ID must be a string")
        self.n = n
        self.position_x = np.random.randint(0,n, dtype= np.int64)
        self.position_y = np.random.randint(0,n, dtype= np.int64)
        self.individual_id = individual_id
        self.f_values = np.zeros([n,n], dtype= np.float64) #Initializes an n x n matrix that will be used with generate_data 
        self.visited = np.zeros([n,n], dtype=np.int64)
        self.visited[self.position_x,self.position_y] = 1
        self.init_config()
        self.costs = self.precompute_costs(self.f_values) 
        i_options = np.array([-1, 0, 1], dtype=np.int16)
        j_options = np.array([-1, 0, 1], dtype =np.int16)
        i_grid, j_grid = np.meshgrid(i_options, j_options)
        self.potential_moves = np.stack((i_grid, j_grid), axis=-1, dtype=np.int16).reshape(-1, 2)
        self.potential_moves = self.potential_moves[np.any(self.potential_moves != [0, 0], axis=1)]
        self.beento = []
        
    def precompute_costs(self, f_values):
        cost_right = np.roll(f_values, shift=-1, axis=1) - f_values
        cost_left = np.roll(f_values, shift=1, axis=1) - f_values
        cost_up = np.roll(f_values, shift=1, axis=0) - f_values    
        cost_down = np.roll(f_values, shift=-1, axis=0) - f_values
        cost_up_left = np.roll(np.roll(f_values, shift=1, axis=0), shift=1, axis=1) - f_values
        cost_up_right = np.roll(np.roll(f_values, shift=1, axis=0), shift=-1, axis=1) - f_values
        cost_down_left = np.roll(np.roll(f_values, shift=-1, axis=0), shift=1, axis=1) - f_values
        cost_down_right = np.roll(np.roll(f_values, shift=-1, axis=0), shift=-1, axis=1) - f_values
        cost_neighbours = (cost_right + cost_left + cost_up + cost_down + cost_up_left + cost_up_right + cost_down_left + cost_down_right) / 8

        return {
            (1, 0): cost_right, 
            (-1, 0): cost_left, 
            (0, 1): cost_down, 
            (0, -1): cost_up, 
            (-1, -1): cost_up_left, 
            (-1, 1): cost_down_left, 
            (1, -1): cost_up_right, 
            (1, 1): cost_down_right,
        } 

    def name(self):
        return 'Grid 7'
    
    def init_config(self):
        self.f_values[:] = generate_data(self.n, self.individual_id)

    def copy(self):
        return deepcopy(self)

    def __repr__(self):
        return str(self.f_values)

    def display(self):
        """
        As there's no clear way on how to display the problem, 
        I have decided to show it as a grid, where the current
        position is highlighted in red and the previous position
        in blue
        """
        plt.imshow(self.visited, cmap='Greys')
        plt.colorbar()
        plt.show()        

    def display_beta(self, beta):
        plt.imshow(self.f_values, cmap='viridis')
        plt.plot(self.position_y, self.position_x, 'ro')
        plt.title(f"Beta = {beta}")
        plt.colorbar()
        plt.show()
    def cost(self):
        return self.f_values[self.position_x, self.position_y]
    def display_end(self):
        plt.imshow(self.f_values, cmap='viridis')
        plt.colorbar()
        
        x_coords, y_coords = zip(*self.beento)

    # Plot a line that follows the visitation order
        plt.plot(y_coords, x_coords, color='k', linewidth=1)

        plt.show()
        

    def propose_move(self):
        """
        This function handles the way a move is proposed.
        It looks at the current position of x and y, initially chosen at random
        in the __init__, and then randomly chooses whether to take a step "forward"
        or "backward" in the x and y direction. It then returns the tuple with the direction
        of the move, formatted as following:

        move = (direction to new x, direction to new y)
        """
        chosen_move_idx = np.random.randint(0,len(self.potential_moves))
        chosen_move = self.potential_moves[chosen_move_idx]
        return chosen_move

    def compute_delta_cost(self, move):
        current_x, current_y = self.position_x, self.position_y
        dx, dy = move
        return self.costs[(dx, dy)][current_x, current_y]

        
    def accept_move(self, move):
        """
        This function handles the way a move is accepted
        For plotting purposes, it updates the previous x and y 
        position to the current one, unpacks the move, and updates
        the current position to the new one
        """
        n = self.n
        current_x, current_y = self.position_x, self.position_y
        dx, dy = move
        new_x, new_y = (current_x + dx) % n, (current_y + dy) % n
        self.old_position_x = current_x
        self.old_position_y = current_y
        self.position_x = new_x
        self.position_y = new_y
        self.visited[new_x, new_y] += 1
        self.beento.append((self.position_x, self.position_y))

        

# %%
