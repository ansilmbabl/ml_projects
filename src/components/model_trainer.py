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
from src.logger  import logging
from src.utils import evaluate_model
from src.utils import save_obj

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")
    
class ModelTrianer:
    def __init__(self):
        self.model_trainer_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self, train_data, test_data):
        try:
            models = {
                    "Random Forest": RandomForestRegressor(),
                    "Decision Tree": DecisionTreeRegressor(),
                    "Gradient Boosting": GradientBoostingRegressor(),
                    "Linear Regression": LinearRegression(),
                    "K-Neighbors Classifier": KNeighborsRegressor(),
                    "XGBClassifier": XGBRegressor(),
                    "CatBoosting Classifier": CatBoostRegressor(verbose=False),
                    "AdaBoost Classifier": AdaBoostRegressor(),
                }
            
            X_train, y_train, X_test, y_test = (
                train_data[:,:-1],
                train_data[:,-1],
                test_data[:,:-1],
                test_data[:,-1]
            )
            logging.info("train and test data split into features and target for model training")
            
            model_report:dict = evaluate_model(X_train, y_train, X_test, y_test, models)
            logging.info("trained on models and returned the r2 score")
            
            best_model_name, best_model_score = max(model_report.items(), key= lambda item: item[1])
            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best model for training and testing is {best_model_name}")
            
            best_model = models[best_model_name]
            save_obj(
                file_path = self.model_trainer_trainer_config.trained_model_file_path,
                obj = best_model
            )
            logging.info(f"saved the model.pkl in {self.model_trainer_trainer_config.trained_model_file_path}")
            
            return r2_score(y_test, best_model.predict(X_test)) 
            
        except Exception as e:
            raise CustomException(e)