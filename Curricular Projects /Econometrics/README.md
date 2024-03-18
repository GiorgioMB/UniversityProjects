# Project Overview
This project was a mandatory assignment for an Econometrics course, led and coded entirely by me as the team lead. It involved a comprehensive analysis of economic and demographic indicators across various countries from 1995 to 2023. The core objective was to examine the influences on GDP per capita by various predictors. Through rigorous statistical modeling, including linear regression, generalized least squares (GLS), and ARIMAX models, we identified significant predictors of GDP per capita and explored the impact of geographical and political factors on economic indicators. The project was recognized for its depth of analysis and clarity of insights, earning full marks in the course.


# Project Structure
1. **Downloading and Loading Libraries**: The script begins by installing and loading necessary R packages for data manipulation, statistical modeling, and visualization.
2. **Data Retrieval**: Economic, demographic, and political indicators are retrieved using the `WDI` package, while democracy data is downloaded from a GitHub repository.
3. **Data Preparation**: The dataset undergoes several preprocessing steps, including merging different data sources, creating dummy variables, and imputing missing values using the `kNN` method.
4. **Data Exploration**: Visualization techniques, such as scatter plots, histograms, and box plots, are employed to explore the relationships between variables and to understand the distribution of key indicators.
5. **Statistical Modeling**: Several models, including linear regression, GLS, and ARIMAX, are fitted to the data to analyze the impact of various predictors on GDP per capita. Model diagnostics and residual analysis are conducted to assess model fit and assumptions.
6. **Results Interpretation**: The results of statistical analyses are presented, offering insights into the significance and impact of different variables on economic performance.

# Libraries (and Languages) Used

- **R Language**: The project is implemented in R, a programming language and environment for statistical computing and graphics.
- **Key R Packages**:
  - `MASS`, `WDI`, `tidyr`, `dplyr`, `VIM`: Data manipulation and preprocessing.
  - `httr`, `jsonlite`, `maps`: Data retrieval and geographical information processing.
  - `lmtest`, `forecast`, `nlme`, `car`: Statistical modeling and diagnostics.
  - `ggplot2`, `metafor`: Data visualization and meta-analysis.
  - `democracyData`: Additional data related to political freedom and democracy status.

This comprehensive project provides valuable insights into global economic and demographic trends, emphasizing the importance of statistical analysis in understanding complex relationships between various indicators.
