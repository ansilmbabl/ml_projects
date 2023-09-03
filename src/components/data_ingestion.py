import os
import pandas as pd
from dataclasses import dataclass
from src.logger import logging  # Import custom logging configuration
from sklearn.model_selection import train_test_split
from src.exception import CustomException  # Import custom exception class
from src.components.data_transformation import DataTransformation

# Define a dataclass to hold configuration paths
@dataclass
class DataIngestionConfig():
    # Define class attributes using default values
    # The @dataclass decorator generates default magic methods such as __init__, __repr__, etc.
    # These methods are created automatically based on the attribute definitions.
    train_data_path = os.path.join("artifacts", "train.csv")
    test_data_path = os.path.join("artifacts", "test.csv")
    raw_data_path = os.path.join("artifacts", "data.csv")

# DataIngestion class responsible for handling data ingestion process
class DataIngestion():
    def __init__(self):
        # Initialize the class with a configuration instance
        self.ingestion_config = DataIngestionConfig()
        
    def dataIngestionInitiate(self):
        # Log entry into the data ingestion process
        logging.info("Entered the data ingestion method or component")
        try:
            # Read the raw data CSV file into a DataFrame
            raw = pd.read_csv("notebook\data\stud.csv")
            logging.info('Read the dataset as dataframe')
            
            # Create the necessary directories to store data files
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            
            # Save the raw data as a CSV file
            raw.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            
            # Initiate train-test split on the raw data
            logging.info("train test split initiated")
            train_data, test_data = train_test_split(raw, test_size=0.2, random_state=12)
            
            # Save the split datasets as CSV files
            train_data.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_data.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            # Log the completion of data ingestion
            logging.info("ingestion of data is completed")
            
            # Return the paths to the created train and test data files
            return (self.ingestion_config.train_data_path, self.ingestion_config.test_data_path)
        
        except Exception as e:
            # If any exception occurs during the process, raise a custom exception
            raise CustomException(e)

# Main execution block
if __name__ == "__main__":
    # Create an instance of the DataIngestion class
    obj = DataIngestion()
    
    # Call the data ingestion method to initiate the process
    train, test = obj.dataIngestionInitiate()
    
    datatransform = DataTransformation()
    datatransform.initiate_data_transformation(train, test)
