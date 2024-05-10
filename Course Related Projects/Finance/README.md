# Project Overview
This project, completed for the Principle of Finance course, involved optimizing an investment portfolio using a Genetic Algorithm. 
+The core objective was to find a portfolio allocation that balanced financial returns, environmental impact (CO2 emissions), and diversification.

# Project Structure
1. **Downloading and Loading Libraries**: The notebook begins by installing and loading the necessary libraries, such as DEAP and pandas.
2. **Data Retrieval**: Data is retrieved from an excel file handed to us for the assignment.
3. **Function and Class Definition**: Functions and classes for generation, validation and fitness checking are defined 
4. **Genetic Algorithm**: A genetic loop using DEAP is defined and is subsequently run to find the optimal allocation
5. **Results Collection and Display**: The portfolio with the best overall fitness (considering all objectives) is identified, the allocation is displayed and portfolio metrics are calculated on the optimised portfolio 
# Libraries Used
  - `pandas`: Used for data manipulation, loading data from Excel, and creating dataframes.
  - `deap`:  Provides tools for implementing a Genetic Algorithm, including individual creation, mutation, crossover, and selection.
  - `numpy`: Used for numerical computations and array manipulation, particularly relevant for portfolio calculations.

This notebook demonstrates how a Genetic Algorithm can be used to optimize a portfolio allocation, balancing financial returns, 
environmental impact, and diversification. The approach allows to find a portfolio that achieves the desired balance between these competing objectives.
