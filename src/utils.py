import os
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException


def save_obj(file_path, obj):
    try:
        # Extract the directory path from the file path
        dir_path = os.path.dirname(file_path)
        
        # Ensure that the directory exists or create it
        os.makedirs(dir_path, exist_ok=True)
        
        # Open the specified file path in binary write mode
        with open(file_path, "wb") as file_obj:
            # Serialize and write the Python object to the file
            pickle.dump(obj, file_obj)
            
    except Exception as e:
        raise CustomException(e)
    
    
def evaluate_model(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}
        for item in models:
            model = models[item]
            param = params[item]
            
            hyper_params = GridSearchCV(
                estimator = model,
                param_grid = param,
                cv = 3,
                verbose=2
            )
            hyper_params.fit(X_train, y_train)
            
            model.set_params(**hyper_params.best_params_)
            model.fit(X_train, y_train)
            
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            train_r2_score = r2_score(y_train, y_train_pred)
            test_r2_score = r2_score(y_test, y_test_pred)
            
            report[item] = test_r2_score
        return report
    
    except Exception as e:
        raise CustomException(e)
