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
from utils import save_obj


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
        
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("read train and test data as dataframes")
            
            target_variable = "math_score"
            input_feature_train_df = train_df.drop(columns = [target_variable], axis = 1)
            target_feature_train_df = train_df[target_variable]
            
            input_feature_test_df = test_df.drop(columns = [target_variable], axis = 1)
            target_feature_test_df = test_df[target_variable]
            logging.info("both testing and training data are converted to features and target")
            
            preprocessing_object = self.get_data_transformer_object()
            logging.info("preprocessing object obteained")
            
            input_feature_train_arr = preprocessing_object.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_object.transform(input_feature_test_df)
            logging.info("applied preprocessing transformations on input features")
            
            save_obj(
                file_path= self.data_transformation_config.preprocessor_obj_file_path,
                obj= preprocessing_object
            )
            logging.info("preprocessing object saved")
            
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            logging.info("returning train and test data after preprocessing")
            return (train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path)
            
        except Exception as e:
            raise CustomException(e)