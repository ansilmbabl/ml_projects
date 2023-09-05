import os
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_model
from src.utils import save_obj

@dataclass
class ModelTrainerConfig:
    # Define configuration for the trained model file path
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_data, test_data):
        try:
            # Define a dictionary of regression models to evaluate
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            
            # Parameters for hyperparmeter tuning
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],

                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }

            }

            # Split the training and testing data into features and target variables
            X_train, y_train, X_test, y_test = (
                train_data[:, :-1],
                train_data[:, -1],
                test_data[:, :-1],
                test_data[:, -1]
            )
            logging.info("Train and test data split into features and target for model training")

            # Evaluate each model's performance and store the results in a dictionary
            model_report: dict = evaluate_model(X_train, y_train, X_test, y_test, models, params=params)
            logging.info("Trained on models using GridSearchCV and returned the R2 score")

            # Find the best-performing model based on the R2 score
            best_model_name, best_model_score = max(model_report.items(), key=lambda item: item[1])

            # If the best model's score is below a threshold, raise an exception
            if best_model_score < 0.6:
                raise CustomException("No best model found")

            logging.info(f"Best model for training and testing is {best_model_name}")

            # Retrieve the best model from the dictionary
            best_model = models[best_model_name]

            # Save the best model to a file
            save_obj(
                file_path=self.model_trainer_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info(f"Saved the model.pkl in {self.model_trainer_trainer_config.trained_model_file_path}")

            # Return the R2 score of the best model on the test data
            return r2_score(y_test, best_model.predict(X_test))

        except Exception as e:
            raise CustomException(e)
