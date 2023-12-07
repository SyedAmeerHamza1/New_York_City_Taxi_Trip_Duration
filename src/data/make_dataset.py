import pandas as pd 
import numpy as np
import os
import sys
import pathlib
import yaml

from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging

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

       df.drop(columns='id', axis=1, inplace=True)

       df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
       df["pickup_hour"] = df["pickup_datetime"].dt.hour
       df["pickup_minute"] = df["pickup_datetime"].dt.minute
       df["pickup_second"] = df["pickup_datetime"].dt.second/100
       df["pickup_minute_of_the_day"] = df["pickup_hour"] * 60 + df["pickup_minute"]
       df["pickup_day_week"] =df["pickup_datetime"].dt.dayofweek
       df["pickup_month"] = df["pickup_datetime"].dt.month


       df["dropoff_datetime"] = pd.to_datetime(df["dropoff_datetime"])
       df["dropoff_hour"] = df["dropoff_datetime"].dt.hour
       df["dropoff_minute"] = df["dropoff_datetime"].dt.minute
       df["dropoff_second"] = df["dropoff_datetime"].dt.second/100
       df["dropoff_minute_of_the_day"] = df["dropoff_hour"] * 60 + df["dropoff_minute"]
       df["dropoff_day_week"] =df["dropoff_datetime"].dt.dayofweek
       df["dropoff_month"] = df["dropoff_datetime"].dt.month

       df.drop(columns=["pickup_datetime", "dropoff_datetime"], axis=1, inplace=True)

       os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

       logging.info("Train Test Split initiated")

       train_df, test_df=train_test_split(df, test_size=test_split, random_state=seed)

       train_df.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
       test_df.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

       logging.info("Ingestion of the Data is completed")
     
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
            
            data_ingestion_instance = DataIngestion()

            data_ingestion_instance.initiate_data_ingestion(data_path=data_path, 
                                                            test_split=params['test_split'], 
                                                            seed=params['seed'])
            
        except Exception as e:
           raise CustomException(e, sys)


if __name__=="__main__":
   DataIngestion.main()
   


    
      
