import pandas as pd 
import numpy as np
import os
import sys
import pathlib
import yaml

from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

from dataclasses import dataclass

@dataclass
class DataIngestionConifg:
   train_data_path:str=os.path.join("data\interim", "train.csv")
   test_data_path:str= os.path.join("data\interim", "test.csv")


class DataIngestion:
   
   
   def __init__(self):
    self.ingestion_config=DataIngestionConifg()


   def initiate_data_ingestion(self, data_path, test_split, seed):
     
     
     
     logging.info("Entered The data ingestion method")

     try:
       
       df= pd.read_csv(data_path)

       logging.info("Read the dataset as dataframe")

       os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

       logging.info("Train Test Split initiated")

       train_set, test_set= train_test_split(df, test_size=test_split, random_state=seed)

       train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
       test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

       logging.info("Ingestion of the Data is completed")

       



       return(
         self.ingestion_config.train_data_path,
         self.ingestion_config.test_data_path
       )
     except Exception as e:
       raise CustomException(e, sys)
     
   def main():
        try:
            curr_dir = pathlib.Path(__file__)
            home_dir = curr_dir.parent.parent.parent
            params_file = home_dir.as_posix() + '/params.yaml'
            params = yaml.safe_load(open(params_file))["make_dataset"]

            input_file = sys.argv[1]
            data_path = home_dir.as_posix() + input_file
            
            #data = DataIngestion.initiate_data_ingestion(data_path)
            data_ingestion_instance = DataIngestion()

            data_ingestion_instance.initiate_data_ingestion( data_path=data_path, test_split=params['test_split'], seed=params['seed'])
            
        except Exception as e:
           raise CustomException(e, sys)


if __name__=="__main__":
   DataIngestion.main()
   


    
      
