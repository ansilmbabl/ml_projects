import os
import pickle
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
