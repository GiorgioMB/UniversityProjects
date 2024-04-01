# Project Overview
This project, completed for a Principle of Finance course, involved optimizing an investment portfolio using a Genetic Algorithm. 
I implemented the entire code myself. The core objective was to find a portfolio allocation that balanced financial returns, environmental impact (CO2 emissions), and diversification.

# Project Structure
1. **Downloading and Loading Libraries**: The script begins by installing and loading the necessary libraries, such as DEAP and pandas.
2. **Data Retrieval**: Data is retrieved from an excel file handed to us for the assignment.
3. **Function and Class Definition**: Functions and classes for generation, validation and fitness checking are defined 
4. **Genetic Algorithm**: A genetic loop using DEAP is defined and is subsequently run to find the optimal allocation
5. **Results Collection and Display**: The individual with the best overall fitness (considering all objectives) is identified, the allocation is displayed and portfolio metrics are calculated on the optimised portfolio 
6. 
# Libraries Used
  - `pandas`
  - `deap`
  - `numpy`

This notebook demonstrates how a Genetic Algorithm can be used to optimize a portfolio allocation, balancing financial returns, 
environmental impact, and diversification. The approach allows to find a portfolio that achieves the desired balance between these competing objectives.
