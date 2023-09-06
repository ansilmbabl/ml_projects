import pandas as pd
from src.exception import CustomException
from src.utils import load_object

# Define a class for a prediction pipeline
class PredictPipeline:
    def predict(self, feature):
        try:
            # Define the file paths for the preprocessor and model artifacts
            preprocessor_path = "artifacts\preprocessor.pkl"
            model_path = "artifacts\model.pkl"
            
            # Load the preprocessor object from the specified file path
            preprocessor = load_object(preprocessor_path)
            
            # Transform the input feature using the preprocessor
            preprocessed_data = preprocessor.transform(feature)
            
            # Load the model object from the specified file path
            model = load_object(model_path)
            
            # Make a prediction using the model and the preprocessed data
            prediction = model.predict(preprocessed_data)
            
            # Return the prediction result
            return prediction[0]
        
        except Exception as e:
            # If an exception occurs, raise a CustomException with the original exception as its cause
            raise CustomException(e)

# Define a class for custom data
class CustomData:
    def __init__(self, 
                gender: str,
                race_ethnicity: str,
                parental_level_of_education,
                lunch: str,
                test_preparation_course: str,
                reading_score: int,
                writing_score: int):
        
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score
        
    def data_to_dataframe(self):
        try:
            # Create a dictionary with custom data attributes
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }
            
            # Convert the dictionary to a pandas DataFrame
            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            # If an exception occurs, raise a CustomData exception with the original exception as its cause
            raise CustomException(e)
