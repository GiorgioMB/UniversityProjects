import datetime
from turtle import pos
from typing import Any
import numpy as np
import optuna
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as tfl
from DataFetcher import DataFetcher
import tensorflow_probability as tfp
from tensorflow.keras.models import Sequential
from LSTM_Helper import ParabolicVGPLayer
tfpl = tfp.layers
tfd = tfp.distributions
tfpk = tfp.math.psd_kernels
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt



def prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    prior_model = Sequential([
        tfpl.DistributionLambda(lambda t: tfd.MultivariateNormalDiag(loc=tf.zeros(n), scale_diag=tf.ones(n))),
    ])
    return prior_model

def posterior(kernel_size, bias_size, dtype=None):

    n = kernel_size + bias_size
    posterior_model = Sequential([
        tfpl.VariableLayer(tfpl.MultivariateNormalTriL.params_size(n), dtype=dtype),
        tfpl.MultivariateNormalTriL(n)
    ])
    return posterior_model


class LSTMModel_Prediction:
    """
    Inputs:
    -ticker: Ticker symbol of the stock to be analyzed
    -window: Length of the window to use for the LSTM model (default: 3)
    -optimization_epochs: Number of epochs to use for optuna's optimization process (default: 50)
    -optimize_all: Whether to optimize all hyperparameters or only the learning rate (default: False)
    -evaluate: Whether to evaluate the model's performance on the test set (default: False)
    -predict: Whether to predict future values of the stock (default: False) - requires days_to_predict to be specified
    -days_to_predict: Number of days to predict (default: None)
    -percentage: Percentage of the data to use for training and testing (default: -1, which uses all data)
    -verbose: Whether to print out progress messages (default: False)

    Outputs:
    -if evaluate == True: Dictionary of evaluation metrics and plot of the predicted against the actual values
    -if predict == True: Numpy array of predicted values
    
    This class is structured to manage the lifecycle of a predictive model from data acquisition and preprocessing to model training and evaluation or prediction.
    Upon initialization, it fetches stock price data for a specified ticker and pre-processes this data into a suitable format for time series analysis. 
    It employs a windowing technique to structure the data, and divides it into training, validation, and testing sets to facilitate model evaluation.
    It utilizes the Optuna library to optimize the learning rate for training the LSTM model, for efficient training and better model performance.
    """

    def __init__(self, ticker, window = 3, optimization_epochs = 50, optimize_all = False, evaluate = False, predict = False, days_to_predict = None, percentage = -1, verbose = False):
        self.percentage = percentage
        self.verbose = verbose
        if predict == True and days_to_predict == None:
            raise ValueError("The number of days to predict must be specified if the model is set to predict is set to True")
        self.df, _ = DataFetcher(ticker).get_historical_data(ticker)
        self.window = window
        self.windowed_dataframe = self.df_to_windowed_df(self.df)
        self.dates, self.X, self.y = self.windowed_df_to_date_X_y(
            self.windowed_dataframe
        )
        (
            self.X_train,
            self.y_train,
            self.X_val,
            self.y_val,
            self.X_test,
            self.y_test,
        ) = self.return_xandy_trainandtest(self.dates, self.X, self.y)
        self.optuna_trials = optimization_epochs
        self.optuna_all = optimize_all
        if evaluate:
          self.evaluate_model()
        if predict:
            self.days = days_to_predict
            self.future_prediction()
    def to_utc(self, timestamp) -> datetime.datetime:
        """
        Converts a timestamp to UTC.

        Input:
            -timestamp: input timestamp
        """
        if self.verbose:
            print("Converting timestamp to UTC")
        if timestamp.tzinfo is None:
            return timestamp.tz_localize("UTC")
        else:
            return timestamp.tz_convert("UTC")

    def str_to_datetime(self, date_str) -> datetime.datetime:
        """
        Converts a string to a datetime object.

        Input:
            -date_str: Datestring in format 'YYYY-MM-DD'
        """
        if self.verbose:
            print("Converting string to datetime object")
        year, month, day = map(int, date_str.split("-"))
        return datetime.datetime(year=year, month=month, day=day)

    def get_recent_data(self, dataframe) -> pd.DataFrame:
        """
        Extracts the most recent data from a dataframe.

        Input:
            -dataframe: input dataframe
        """
        if self.verbose:
            print("Extracting recent data")
        percentage = self.percentage
        if dataframe.index.tz is not None:
            dataframe.index = dataframe.index.tz_localize(None)
        df_sorted = dataframe.sort_index(ascending=False)
        top_percent = int(len(df_sorted) * (percentage / 100))
        recent_data = df_sorted.head(top_percent)
        return recent_data.sort_index(ascending=True)

    def df_to_windowed_df(self, dataframe) -> pd.DataFrame:
        """
        Transforms a dataframe into a windowed dataframe for time series analysis.

        Input:
            -dataframe: input dataframe
        """
        if self.verbose:
            print("Transforming dataframe into windowed dataframe")
        n = self.window
        if dataframe.index.tz is not None:
            dataframe.index = dataframe.index.tz_localize(None)
        recent_df = self.get_recent_data(dataframe) if self.percentage > 0 else dataframe
        first_date_str = str(recent_df.index[self.window].date())
        last_date_str = str(recent_df.index.max().date())
        first_date = self.str_to_datetime(first_date_str)
        last_date = self.str_to_datetime(last_date_str)
        target_date = first_date
        dates = []
        X, Y = [], []
        last_time = False
        while True:
            df_subset = dataframe.loc[:target_date].tail(n + 1)
            if len(df_subset) != n + 1:
                print(f"Error: Window of size {n} is too large for date {target_date}")
            values = df_subset["Close"].to_numpy()
            x, y = values[:-1], values[-1]
            dates.append(target_date)
            X.append(x)
            Y.append(y)
            next_week = dataframe.loc[
                target_date : target_date + datetime.timedelta(days=7)
            ]
            next_datetime_str = str(next_week.head(2).tail(1).index.values[0])
            next_date_str = next_datetime_str.split("T")[0]
            next_date = self.str_to_datetime(next_date_str)
            if last_time:
                break
            target_date = next_date
            if target_date == last_date:
                last_time = True
        ret_df = pd.DataFrame({})
        ret_df["Target Date"] = dates
        X = np.array(X)
        for i in range(0, n):
            ret_df[f"Target-{n-i}"] = X[:, i]
        ret_df["Target"] = Y
        return ret_df

    def windowed_df_to_date_X_y(self, windowed_dataframe) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Splits a windowed dataframe into dates, feature matrix X, and target vector y.

        Input:
            -windowed_dataframe: input windowed dataframe
        """
        if self.verbose:
            print("Splitting windowed dataframe into dates, X, and y")
        df_as_np = windowed_dataframe.to_numpy()
        dates = df_as_np[:, 0]
        middle_matrix = df_as_np[:, 1:-1]
        X = middle_matrix.reshape((len(dates), middle_matrix.shape[1], 1))
        Y = df_as_np[:, -1]
        return dates, X.astype(np.float32), Y.astype(np.float32)


    def return_xandy_trainandtest(self, dates, X, y) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Splits data into training, validation and test sets.

        Input:
            -dates: Array of dates
            -X: Feature matrix
            -y: Target vector
        """
        if self.verbose:
            print("Splitting data into training, validation, and test sets")
        q_80 = int(len(dates) * 0.8)
        q_90 = int(len(dates) * 0.9)
        dates_train, X_train, y_train = (
            dates[:q_80],
            X[:q_80],
            y[:q_80],
        )  ##dates_train is not used now
        dates_val, X_val, y_val = (
            dates[q_80:q_90],
            X[q_80:q_90],
            y[q_80:q_90],
        )  ##dates_val is not used now
        dates_test, X_test, y_test = (
            dates[q_90:],
            X[q_90:],
            y[q_90:],
        )  ##dates_test is not used now
        return X_train, y_train, X_val, y_val, X_test, y_test

    def find_best(self, X_train, y_train, X_test, y_test, n_trials, optimize) -> tuple[float, int, Any, Any]:
        """
        Utilizes Optuna to find the optimal learning rate for the model.
        Inputs:
            -X_train, y_train: Training data
            -X_test, y_test: Testing data
            -n_trials: Number of trials for Optuna
            -optimize: Boolean value to determine whether to optimize the number of epochs and batch size
        Outputs:
            -best_lr: Optimal learning rate
            -best_points: Optimal number of inducing points 
            -best_epochs: Optimal number of epochs (if optimize == True)
            -best_batch: Optimal batch size (if optimize == True)
        """
        if self.verbose:
            print("Finding optimal hyperparameters...")
        if optimize == True:
            def objective(trial) -> float:
                """
                Defines the objective function for Optuna.
                Inputs:
                    -trial: Optuna trial object
                Outputs:
                    -Final validation loss
                """
                if self.verbose:
                    print("Optimization in progress, {}".format(trial))
                try:
                    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
                    ip = trial.suggest_int("ip", 10, 50)
                    epoch = trial.suggest_int("epoch", 10, 500)
                    batch = trial.suggest_categorical("batch", [16, 32, 64, 128])
                    model = self.optuna_build_model(learning=lr, ind_points=ip)
                    history = model.fit(
                        X_train, 
                        y_train, 
                        validation_data=(X_test, y_test), 
                        epochs=epoch, 
                        batch_size=batch, 
                        verbose=0
                    )
                    return history.history['val_loss'][-1]
                except Exception as e:
                    print(f"Exception {type(e): e} encountered at trial: {trial}. Skipping iteration")
                    return np.inf
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=n_trials)
            best_lr = study.best_params['lr']
            best_points = study.best_params['ip']
            best_epochs = study.best_params['epoch']
            best_batch = study.best_params['batch']
            return best_lr, best_points, best_epochs, best_batch
        else:
            def objective(trial) -> float:
                """
                Defines the objective function for Optuna.
                Inputs:
                    -trial: Optuna trial object
                Outputs:
                    -Final validation loss
                """
                if self.verbose:
                    print("Optimization in progress, {}".format(trial))
                try:
                    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
                    ip = trial.suggest_int("ip", 10, 50)
                    model = self.optuna_build_model(learning=lr, ind_points=ip)
                    history = model.fit(
                        X_train, 
                        y_train, 
                        validation_data=(X_test, y_test), 
                        epochs=100, 
                        batch_size=32, 
                        verbose=0
                    )
                    return history.history['val_loss'][-1]
                except Exception as e:
                    print(f"Exception {type(e): e} encountered at trial: {trial}. Skipping iteration")
                    return np.inf
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=n_trials)
            best_lr = study.best_params['lr']
            best_points = study.best_params['ip']
            return best_lr, best_points, None, None

    def optuna_build_model(self, learning, ind_points) -> Sequential:
        """
        Builds the Bayesian LSTM model for Optuna input.
        Input:
            -learning_rate: Learning rate passed from the optimizer
        """
        if self.verbose:
            print("Building model for Optuna optimization")
        model = tf.keras.Sequential([tfl.Input((self.window, 1)), tfl.LSTM(64)])
        for i in range(self.window):
            model.add(tfpl.DenseVariational(32, posterior, prior, activation="relu"))
        model.add(tfpl.VariationalGaussianProcess(
                    num_inducing_points=ind_points, 
                    kernel_provider=ParabolicVGPLayer(1,1),
            ))
        model.add(
            tfpl.DenseVariational(
                1,
                posterior,
                prior,
                activation = "linear"
            )
        )
        model.compile(
            loss="mse",
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning),
            metrics=["mean_absolute_error"],
        )
        return model

    def build_model(self) -> tuple[Sequential, int, int]:
        """
        Builds the Bayesian LSTM model with the optimal learning rate
        Outputs:
            -model: Untrained TensorFlow model object
            -epochs: Number of epochs for training
            -batch_size: Size of the training batch
        """
        if self.verbose:
            print("Building model")
        learning, num_points, epochs, batch = self.find_best(
            self.X_train, self.y_train, self.X_test, self.y_test, self.optuna_trials, self.optuna_all
        )

        model = tf.keras.Sequential([tfl.Input((self.window, 1)), tfl.LSTM(64)])
        for i in range(self.window):
            model.add(tfpl.DenseVariational(32, posterior, prior, activation="relu"))
        model.add(tfpl.VariationalGaussianProcess(
                    num_inducing_points=num_points, 
                    kernel_provider=ParabolicVGPLayer(1,1),
            ))
        model.add(
            tfpl.DenseVariational(
                1,
                posterior,
                prior,
                activation = "linear"
            )
        )
        model.compile(
            loss="mse",
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning),
            metrics=["mean_absolute_error"],
        )
        return model, epochs, batch

    def train_model(self, epochs=100, batch_size=32):
        """
        Trains the Bayesian LSTM model.
        Inputs:
            -model: Untrained TensorFlow model object
            -epochs: Number of epochs for training
            -batch_size: Size of the training batch
        Note:
            -if self.optuna_all is True, the model will be trained with the optimal number of epochs and batch size
        """
        if self.verbose:
            print("Training model")
        model, epo, batch = self.build_model()
        if epo is not None:
          epochs = epo
          batch_size = batch
        X_train, y_train, X_val, y_val = self.X_train, self.y_train, self.X_val, self.y_val
        try:
            history = model.fit(
                X_train, 
                y_train, 
                validation_data=(X_val, y_val), 
                epochs=epochs, 
                batch_size=batch_size, 
            )
            
            return model, history
        except Exception as e:
            print(f"An error occurred during model training: {e}")
            return None, None
    def evaluate_model(self) -> dict[str, float]:
        """
        Evaluates the trained Bayesian LSTM model on the test set.
        Outputs:
            -mse: Mean squared error
            -mae: Mean absolute error
            -rmse: Root mean squared error
            -r_squared: R-squared score
            -directional_accuracy: Directional accuracy
        """
        if self.verbose:
            print("Evaluating model")
        model, _ = self.train_model()
        X_test, y_test = self.X_Test, self.y_test
        try:
            y_pred = model.predict(X_test).flatten()
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r_squared = r2_score(y_test, y_pred)
            directional_accuracy = np.mean(
                np.sign(y_test[1:] - y_test[:-1]) == np.sign(y_pred[1:] - y_pred[:-1])
            ) * 100
            plt.figure(figsize=(10, 6))
            plt.plot(y_test, label='Actual')
            plt.plot(y_pred, label='Predicted')
            plt.title('Model Evaluation')
            plt.xlabel('Time')
            plt.ylabel('Stock Price')
            plt.legend()
            plt.show()

            return {
                "mean_squared_error": mse,
                "mean_absolute_error": mae,
                "root_mean_squared_error": rmse,
                "r_squared": r_squared,
                "directional_accuracy": directional_accuracy
            }
        except Exception as e:
            print(f"An error occurred during model evaluation: {e}")
            return {}
    def future_prediction(self) -> np.ndarray:
        """
        Predicts the stock price for a specified number of days into the future.
        Outputs:
            -predictions: Array of predicted stock prices
        """ 
        if self.verbose:
            print("Predicting future stock prices")
        steps = self.days
        model, _ = self.train_model()
        predictions = np.zeros(steps)

        for i in range(steps):
            X_test = self.X_test[-1:]
            y_pred = model.predict(X_test).flatten()
            predictions[i] = y_pred[-1]
            new_window = np.append(X_test[:, 1:, :], [[y_pred[-1]]]).reshape(1, self.window, 1)
            self.X_test = np.append(self.X_test, new_window, axis=0)
            if i % 3 == 0:
                if self.verbose:
                    print(f"Predicted {i} days into the future, retraining model")
                model.fit(self.X_test, self.y_test, epochs=10, verbose=0)
        return predictions
