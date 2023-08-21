import os
import pandas as pd
from dataclasses import dataclass
from src.logger import logging
from sklearn.model_selection import train_test_split
from src.exception import CustomException


@dataclass
class DataIngestionConfig():
    train_data_path = os.path.join("artifacts","train.csv")
    test_data_path = os.path.join("artifacts","test.csv")
    raw_data_path = os.path.join("artifacts","data.csv")
    
    
class DataIngestion():
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
    def dataIngestionInitiate(self):
        logging.info("Entered the data ingestion method or component")
        try:
            raw = pd.read_csv("notebook\data\stud.csv")
            logging.info('Read the dataset as dataframe')
            
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            
            raw.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            
            logging.info("train test split initiated")
            train_data, test_data = train_test_split(raw, test_size=0.2, random_state=12)
            
            train_data.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_data.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logging.info("ingestion of data is completed")
            
            return (self.ingestion_config.train_data_path, self.ingestion_config.test_data_path)
        
        except Exception as e:
            raise CustomException(e)
        
        
        
if __name__ == "__main__":
    obj = DataIngestion()
    obj.dataIngestionInitiate()