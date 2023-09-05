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
        # Initialize an empty dictionary to store evaluation results
        report = {}
        
        # Loop through each model specified in the 'models' dictionary
        for item in models:
            model = models[item]  # Get the model
            param = params[item]  # Get the hyperparameter grid
            
            # Create a GridSearchCV instance for hyperparameter tuning
            hyper_params = GridSearchCV(
                estimator=model,     # Specify the model
                param_grid=param,   # Specify the hyperparameter grid
                cv=3,               # Number of cross-validation folds
                verbose=2           # Verbosity level (2 for detailed output)
            )
            
            # Fit the grid search to the training data to find the best hyperparameters
            hyper_params.fit(X_train, y_train)
            
            # Set the model's hyperparameters to the best values found by GridSearchCV
            model.set_params(**hyper_params.best_params_)
            
            # Fit the model with the best hyperparameters on the training data
            model.fit(X_train, y_train)
            
            # Make predictions on both training and testing data
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Compute R-squared scores for both training and testing data
            train_r2_score = r2_score(y_train, y_train_pred)
            test_r2_score = r2_score(y_test, y_test_pred)
            
            # Store the test R-squared score in the report dictionary
            report[item] = test_r2_score
        
        # Return the evaluation report containing test R-squared scores for each model
        return report
    
    except Exception as e:
        # If an exception occurs, raise a custom exception with the error message
        raise CustomException(e)
