import numpy as np
import pandas as pd
import os

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from utils import save_obj

# Define a data class to store configuration parameters
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            # Define numerical and categorical columns
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            # Define a pipeline for numerical columns
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ('scaler', StandardScaler())
                ]
            )

            logging.info("numerical pipeline: Simple imputer, Standard scaler")

            # Define a pipeline for categorical columns
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='most_frequent')),
                    ("encoding", OneHotEncoder()),
                    ("scalar", StandardScaler())
                ]
            )

            logging.info("categorical pipeline: Simple imputer, One-hot encoder, Standard scaler")

            # Create a ColumnTransformer to apply different transformers to different columns
            preprocessor = ColumnTransformer(
                transformers=[
                    ("numerical pipeline", num_pipeline, numerical_columns),
                    ("categorical pipeline", cat_pipeline, categorical_columns)
                ]
            )

            logging.info("preprocessing completed")
            return preprocessor

        except Exception as e:
            raise CustomException(e)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Read training and testing data as DataFrames
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("read train and test data as dataframes")

            target_variable = "math_score"

            # Separate input features and target features for training data
            input_feature_train_df = train_df.drop(columns=[target_variable], axis=1)
            target_feature_train_df = train_df[target_variable]

            # Separate input features and target features for testing data
            input_feature_test_df = test_df.drop(columns=[target_variable], axis=1)
            target_feature_test_df = test_df[target_variable]
            logging.info("both testing and training data are converted to features and target")

            # Get the data transformer object
            preprocessing_object = self.get_data_transformer_object()
            logging.info("preprocessing object obtained")

            # Apply preprocessing transformations to input features for both training and testing data
            input_feature_train_arr = preprocessing_object.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_object.transform(input_feature_test_df)
            logging.info("applied preprocessing transformations on input features")

            # Save the preprocessing object to a file
            save_obj(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_object
            )
            logging.info("preprocessing object saved")

            # Combine input features and target features for both training and testing data
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("returning train and test data after preprocessing")
            return (train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path)

        except Exception as e:
            raise CustomException(e)
