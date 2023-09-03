import numpy as np
import pandas as pd
import os

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataTransofrmationCofig:
    preprocessor_obj_file_path = os.path.join("artifacts", 'preprocessor.pkl')
    
    
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransofrmationCofig()
        
    def get_data_transformer_object(self):
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]
            
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ('scaler', StandardScaler())
                ]
            )
            
            logging.info("numerical pipline : simple imputer, standard scalar")
            
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='mode'))
                    ("encoding", OneHotEncoder()),
                    ("scalar", StandardScaler())
                ]
            )
            
            logging.info("categoerical pipline : simple imputer,One hot encoder, standard scalar")
            
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