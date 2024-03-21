listofpackages <- c(
  "MASS",
  "WDI",
  "tidyr",
  "dplyr",
  "VIM",
  "httr",
  "jsonlite",
  "lmtest",
  "forecast",
  "nlme",
  "car",
  "ggplot2",
  "metafor",
  "maps",
  "tseries"
)
if (!require("democracyData")) {
  remotes::install_github("xmarquez/democracyData")
}
newpackages <- listofpackages[!(listofpackages %in% installed.packages()[,"Package"])]
if(length(newpackages)) install.packages(newpackages,
                                         dependencies = TRUE,
                                         repos = "http://cran.us.r-project.org"
)
library(MASS)
library(WDI)
library(tidyr)
library(dplyr)
library(VIM)
library(httr)
library(jsonlite)
library(lmtest)
library(forecast)
library(nlme)
library(car)
library(ggplot2)
library(metafor)
library(democracyData)
library(maps)
library(tseries)
start_date <- 1995
end_date <- 2023

gdp_per_capita <- WDI(country = "all",
                      "NY.GDP.PCAP.CD",
                      start = start_date,
                      end = end_date
)

saving_rate <- WDI(country = "all",
                   "NY.GNS.ICTR.ZS",
                   start = start_date,
                   end = end_date
)

population_growth <- WDI(country = "all",
                         "SP.POP.GROW",
                         start = start_date,
                         end = end_date
)

fertility <- WDI(country = "all",
                 "SP.DYN.TFRT.IN",
                 start = start_date,
                 end = end_date
)

co2_emission <- WDI(country = "all",
                    "EN.ATM.CO2E.PC",
                    start = start_date,
                    end = end_date
)

pol_stability_lower <- WDI(country = "all",
                           "PV.PER.RNK.LOWER",
                           start = start_date,
                           end = end_date
)

pol_stability_upper <- WDI(country = "all",
                           "PV.PER.RNK.UPPER",
                           start = start_date,
                           end = end_date
)

research <- WDI(country = "all",
                "GB.XPD.RSDV.GD.ZS",
                start = start_date,
                end = end_date
)

dem_data  <- download_fh()

world_map <- map_data("world")

print("Downloaded the dataset")
dem_data <- dem_data %>% select(fh_country, year, status)
dem_data$dummy_PF <- ifelse(dem_data$status == "PF", 1, 0)
dem_data$dummy_F <- ifelse(dem_data$status == "F", 1, 0)
colnames(dem_data)[1] <- "country"
dem_data$status <- NULL
country_centroids <- aggregate(
  cbind(long, lat) ~ region,
  data = world_map,
  FUN = function(x) median(range(x))
)
colnames(country_centroids) <- c("country", "longitude", "latitude")
country_centroids$longitude <- NULL
country_centroids$dummy_30_60 = ifelse(
  abs(country_centroids$latitude) >= 30 & abs(country_centroids$latitude) < 60,
  1,
  0
)
country_centroids$dummy_60_plus = ifelse(
  abs(country_centroids$latitude) >= 60,
  1,
  0
)
country_centroids$latitude <- NULL
data_regression <- merge.data.frame(gdp_per_capita, saving_rate)
data_regression <- merge.data.frame(data_regression, population_growth)
data_regression <- merge.data.frame(data_regression, co2_emission)
data_regression <- merge.data.frame(data_regression, fertility)
data_regression <- merge.data.frame(data_regression, pol_stability_lower)
data_regression <- merge.data.frame(data_regression, research)
data_regression <- merge.data.frame(data_regression, pol_stability_upper)
data_regression <- merge.data.frame(data_regression, country_centroids)
last_year_observed <- max(data_regression$year)
dem_data <- dem_data %>% filter(year >= start_date & year <= last_year_observed)
data_regression <- merge.data.frame(data_regression, dem_data)
subsetted_data_regression = subset(data_regression, year == last_year_observed)
in_subset <- data_regression$country %in% subsetted_data_regression$country
data_regression <- data_regression[in_subset, ]
print("Merged the dataset")
print(paste("Number of unique countries:", length(unique(data_regression$country))))
print(unique(data_regression$country))
data_regression <- kNN(data_regression, k = 10)
summary(data_regression)
weighted_average <- function(lower, upper) {
  weight_l <- 0.5 + (lower / 200)
  weight_u <- 0.5 + ((100 - upper) / 200)
  rtv <- (lower * weight_l + upper * weight_u) / (weight_l + weight_u)
  return(rtv)
}
data_regression$pol_stability <- mapply(
  weighted_average,
  data_regression$PV.PER.RNK.LOWER,
  data_regression$PV.PER.RNK.UPPER
)
data_regression$gdp <- as.numeric(data_regression$NY.GDP.PCAP.CD)
data_regression$saving <- as.numeric(data_regression$NY.GNS.ICTR.ZS)
data_regression$pop_growth <- as.numeric(data_regression$SP.POP.GROW)
data_regression$fertility <- as.numeric(data_regression$SP.DYN.TFRT.IN)
data_regression$co2_emission <- as.numeric(data_regression$EN.ATM.CO2E.PC)
data_regression$pol_stability <- as.numeric(data_regression$pol_stability)
data_regression$research <- as.numeric(data_regression$GB.XPD.RSDV.GD.ZS)
data_regression$iso2c <- NULL
data_regression$iso3c <- NULL
data_regression$NY.GDP.PCAP.CD <- NULL
data_regression$NY.GNS.ICTR.ZS <- NULL
data_regression$SP.POP.GROW <- NULL
data_regression$SP.DYN.TFRT.IN <- NULL
data_regression$EN.ATM.CO2E.PC <- NULL
data_regression$PV.PER.RNK.LOWER <- NULL
data_regression$PV.PER.RNK.UPPER <- NULL
data_regression$GB.XPD.RSDV.GD.ZS <- NULL
data_regression$country_imp <- NULL
data_regression$NY.GDP.PCAP.CD_imp <- NULL
data_regression$NY.GNS.ICTR.ZS_imp <- NULL
data_regression$SP.POP.GROW_imp <- NULL
data_regression$SP.DYN.TFRT.IN_imp <- NULL
data_regression$EN.ATM.CO2E.PC_imp <- NULL
data_regression$iso2c_imp <- NULL
data_regression$iso3c_imp <- NULL
data_regression$year_imp <- NULL
data_regression$PV.PER.RNK.LOWER_imp <- NULL
data_regression$PV.PER.RNK.UPPER_imp <- NULL
data_regression$GB.XPD.RSDV.GD.ZS_imp <- NULL
data_regression$dummy_F_imp <- NULL
data_regression$dummy_PF_imp <- NULL
data_regression$dummy_30_60_imp <- NULL
data_regression$dummy_60_plus_imp <- NULL
data_regression$gdp <- log(data_regression$gdp)
data_regression$saving <- log(data_regression$saving)
data_regression$pop_growth <- log(data_regression$pop_growth)
data_regression$co2_emission <- log(data_regression$co2_emission)
data_regression$fertility <- log(data_regression$fertility)
data_regression$research <- log(data_regression$research)
data_regression$co2_emission  <- ifelse(
  is.infinite(
    data_regression$co2_emission
  ),
  NA,
  data_regression$co2_emission
)
data_regression$fertility  <- ifelse(
  is.infinite(
    data_regression$fertility
  ),
  NA,
  data_regression$fertility
)
data_regression$research  <- ifelse(
  is.infinite(
    data_regression$research
  ),
  NA,
  data_regression$research
)
data_regression <- kNN(data_regression, k = 10)
data_regression$country_imp <- NULL
data_regression$saving_imp <- NULL
data_regression$pop_growth_imp <- NULL
data_regression$gdp_imp <- NULL
data_regression$year_imp <- NULL
data_regression$co2_emission_imp <- NULL
data_regression$fertility_imp <- NULL
data_regression$pol_stability_imp <- NULL
data_regression$research_imp <- NULL
data_regression$dummy_PF_imp <- NULL
data_regression$dummy_F_imp <- NULL
data_regression$dummy_30_60_imp <- NULL
data_regression$dummy_60_plus_imp <- NULL
print(colnames(data_regression))
print(summary(data_regression))
ggplot(data_regression, aes(x = saving, y = gdp)) +
  geom_point() +
  labs(title = "Scatter plot of GDP and Savings",
       x = "Savings",
       y = "GDP")
ggplot(data_regression, aes(x = pop_growth, y = gdp)) +
  geom_point() +
  labs(title = "Scatter plot of GDP and Population Growth",
       x = "Population Growth",
       y = "GDP")
ggplot(data_regression, aes(x = co2_emission, y = gdp)) +
  geom_point() +
  labs(title = "Scatter plot of GDP and CO2 Emission",
       x = "CO2 Emission",
       y = "GDP")
ggplot(data_regression, aes(x = fertility, y = gdp)) +
  geom_point() +
  labs(title = "Scatter plot of GDP and Fertility",
       x = "Fertility",
       y = "GDP")
ggplot(data_regression, aes(x = pol_stability, y = gdp)) +
  geom_point() +
  labs(title = "Scatter plot of GDP and Political Stability",
       x = "Political Stability",
       y = "GDP")
ggplot(data_regression, aes(x = research, y = gdp)) +
  geom_point() +
  labs(title = "Scatter plot of GDP and Research",
       x = "Research",
       y = "GDP")
model <- lm(gdp ~ saving + pop_growth, data = data_regression)
summary(model)
bptest(model)
plot(model)
model <- lm(gdp ~ co2_emission + fertility + pol_stability + research + pop_growth + dummy_F + dummy_PF + dummy_60_plus + dummy_30_60, data = data_regression)
summary(model)
bptest(model)
plot(model)
residuals <- residuals(model)
acf(residuals, main="ACF of Residuals")
pacf(residuals, main="PACF of Residuals")
Box.test(residuals, type = "Ljung-Box")
gls_model <- gls(gdp ~ pol_stability + co2_emission + research + fertility + pop_growth + dummy_F + dummy_PF + dummy_30_60 + dummy_60_plus, 
                 data = data_regression,
                 corr = corAR1(),
                 control = glsControl(
                   maxIter = 1000,
                   msMaxIter = 100,
                   returnObject = TRUE,
                   tolerance = 1e-6,
                   msTol = 1e-6
                 ),
                 verbose = TRUE
)
summary(gls_model)
data_for_ggplot <- data.frame(
  Fitted = fitted(gls_model),
  Residuals = residuals(gls_model)
)
ggplot(data_for_ggplot, aes(x = Fitted, y = Residuals)) +
  geom_point(alpha = 0.5) + 
  geom_smooth(aes(y = Residuals), method = "loess", formula = 'y ~ x', color = "red") + 
  geom_hline(yintercept = 0, color = "green", linetype = "dashed", size = 1) +
  labs(x = "Fitted Values", y = "Residuals", title = "Fitted vs. Residuals") +
  theme_minimal()
ggplot(mapping = aes(sample = residuals(gls_model))) +
  stat_qq() +
  stat_qq_line() +
  labs(title = "QQ Plot of Residuals", x = "Theoretical Quantiles", y = "Sample Quantiles") +
  theme_minimal()
std_resid <- resid(gls_model, type = "pearson") / sd(resid(gls_model, type = "pearson"))
data_for_plot <- data.frame(
  Fitted = fitted(gls_model),
  SqrtAbsStdResid = sqrt(abs(std_resid))
)
ggplot(data_for_plot, aes(x = Fitted, y = SqrtAbsStdResid)) +
  geom_point(alpha = 0.5) + 
  geom_smooth(aes(y = SqrtAbsStdResid), method = "loess", formula = 'y ~ x', color = "red") + 
  labs(x = "Fitted values", y = "Sqrt(|Standardized residuals|)", title = "Scale-Location Plot") +
  theme_minimal()
residuals <- residuals(gls_model)
acf(residuals, main="ACF of Residuals")
pacf(residuals, main="PACF of Residuals")
Box.test(residuals, type = "Ljung-Box")
decorr_model <- lm(fertility ~ dummy_30_60 + co2_emission, data = data_regression)
data_regression$decorr_fertility <- residuals(decorr_model)
gls_model_2 <- gls(gdp ~ pol_stability + co2_emission + research + dummy_F + dummy_30_60 + dummy_60_plus + decorr_fertility, 
                   data = data_regression,
                   corr = corAR1(),
                   weights = varPower(form = ~fitted(.)),
                   control = glsControl(
                     maxIter = 1000,
                     msMaxIter = 100,
                     returnObject = TRUE,
                     tolerance = 1e-6,
                     msTol = 1e-6
                   ),
                   verbose = TRUE
)
summary(gls_model_2)
data_for_ggplot <- data.frame(
  Fitted = fitted(gls_model),
  Residuals = residuals(gls_model_2)
)
ggplot(data_for_ggplot, aes(x = Fitted, y = Residuals)) +
  geom_point(alpha = 0.5) + 
  geom_smooth(aes(y = Residuals), method = "loess", formula = 'y ~ x', color = "red") + 
  geom_hline(yintercept = 0, color = "green", linetype = "dashed", size = 1) +
  labs(x = "Fitted Values", y = "Residuals", title = "Fitted vs. Residuals") +
  theme_minimal()
ggplot(mapping = aes(sample = residuals(gls_model_2))) +
  stat_qq() +
  stat_qq_line() +
  labs(title = "QQ Plot of Residuals", x = "Theoretical Quantiles", y = "Sample Quantiles") +
  theme_minimal()
std_resid <- resid(gls_model_2, type = "pearson") / sd(resid(gls_model_2, type = "pearson"))
data_for_plot <- data.frame(
  Fitted = fitted(gls_model_2),
  SqrtAbsStdResid = sqrt(abs(std_resid))
)
ggplot(data_for_plot, aes(x = Fitted, y = SqrtAbsStdResid)) +
  geom_point(alpha = 0.5) + 
  geom_smooth(aes(y = SqrtAbsStdResid), method = "loess", formula = 'y ~ x', color = "red") + 
  labs(x = "Fitted values", y = "Sqrt(|Standardized residuals|)", title = "Scale-Location Plot") +
  theme_minimal()
residuals <- residuals(gls_model_2)
acf(residuals, main="ACF of Residuals")
pacf(residuals, main="PACF of Residuals")
Box.test(residuals, type = "Ljung-Box")
adf.test(data_regression$gdp)
adf.test(data_regression$pol_stability)
adf.test(data_regression$co2_emission)
adf.test(data_regression$research)
adf.test(data_regression$decorr_fertility)
gdp_ts <- ts(data_regression$gdp)
exog_vars <- as.matrix(data_regression[, c("pol_stability", "co2_emission", "research", "dummy_F", "dummy_30_60", "dummy_60_plus", "decorr_fertility")])
arimax_model <- auto.arima(gdp_ts, xreg = exog_vars, max.p = 7, max.q = 7, max.P = 7, max.Q = 7, max.order = 21, max.d = 7, max.D = 7, lambda = "auto")
summary(arimax_model)
coefficients <- coef(arimax_model)
var_coefficients <- summary(arimax_model)$var.coef
std_errors <- sqrt(diag(var_coefficients))
t_stats <- coefficients / std_errors
degrees_of_freedom <- length(residuals(arimax_model)) - length(coefficients) # degrees of freedom approximation
p_values <- 2 * pt(-abs(t_stats), df=degrees_of_freedom)
results <- data.frame(Coefficients = coefficients,
                      StdError = std_errors,
                      TStatistic = t_stats,
                      PValue = p_values)
print(results)
data_for_ggplot <- data.frame(
  Fitted = fitted(arimax_model),
  Residuals = residuals(arimax_model)
)
ggplot(data_for_ggplot, aes(x = Fitted, y = Residuals)) +
  geom_point(alpha = 0.5) +
  geom_smooth(aes(y = Residuals),
              method = "loess",
              formula = "y ~ x",
              color = "red"
  ) +
  geom_hline(yintercept = 0, color = "green", linetype = "dashed", size = 1) +
  labs(x = "Fitted Values", y = "Residuals", title = "Fitted vs. Residuals") +
  theme_minimal()
std_resid <- resid(arimax_model) / sd(resid(arimax_model))
data_for_plot <- data.frame(
  Fitted = fitted(gls_model_2),
  SqrtAbsStdResid = sqrt(abs(std_resid))
)

ggplot(data_for_plot, aes(x = Fitted, y = SqrtAbsStdResid)) +
  geom_point(alpha = 0.5) +
  geom_smooth(aes(y = SqrtAbsStdResid),
              method = "loess",
              formula = "y ~ x",
              color = "red"
  ) + 
  labs(x = "Fitted values",
       y = "Sqrt(|Standardized residuals|)",
       title = "Scale-Location Plot"
  ) +
  theme_minimal()
residuals <- residuals(arimax_model)
acf(residuals, main = "ACF of Residuals")
pacf(residuals, main = "PACF of Residuals")
Box.test(residuals, type = "Ljung-Box")