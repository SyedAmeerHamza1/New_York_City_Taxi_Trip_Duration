import pandas as pd
import numpy as np
import os
import sys

from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path= os.path.join("data\processed", "preprocessor.pkl")
    train_arr_path= os.path.join("data\processed", "train_arr.csv")
    test_arr_path= os.path.join("data\processed", "test_arr.csv")

class DataTransformation:
    def __init__(self) -> None:
        self.data_transformation_config= DataTransformationConfig()


    def get_data_transformer_obj(self):
        '''
        This function is responsible for data transformation
        '''

        try:
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


            Num_columns=df.columns[(df.dtypes == float) | (df.dtypes == int)]
            Cate_columns=df.columns[df.dtypes=="object"]

            store_and_fwd_flag=df[["store_and_fwd_flag"]].columns

            logging.info("Pipeline for Nominal encoding")
            trf1= Pipeline(steps=[
                ("ohe",OneHotEncoder(drop='first', dtype=np.int32))
            ])

            logging.info("Pipeline for Standerdscaling")
            trf2= Pipeline(steps=[
                ("SS", StandardScaler())
            ])

            logging.info("Column Transformation")
            preprocessor= ColumnTransformer(
                [
                    ("ohe", trf1, store_and_fwd_flag),
                    ("SS", trf2, Num_columns)
                ], remainder="passthrough")
        except Exception as e:
            raise CustomException(e, sys)
        
    
    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df= pd.read_csv(test_path)

            logging.info("Read train and test data")

            logging.info(f"train DataFrame's head:\n {train_df.head().to_string()}")
            logging.info(f"test DataFrame's head:\n {test_df.head().to_string()}")

            preprocessing_obj= self.get_data_transformer_obj()

            target_col_name='trip_duration'
            
            input_features_train_df= train_df.drop(target_col_name, axis=1)
            target_feature_train_df= train_df[target_col_name]

            
            input_features_test_df= test_df.drop(target_col_name, axis=1)
            target_feature_test_df= test_df[target_col_name]

            input_feature_train_arr= preprocessing_obj.fit_transform(input_features_train_df)
            input_feature_test_arr= preprocessing_obj.transform(input_features_test_df)

            logging.info("applying preprocessing object on training and testing dataset")

            train_arr= np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr= np.c_[input_feature_test_arr, np.array(target_feature_test_df)]


            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            save_object(
                file_path=self.data_transformation_config.train_arr_path,
                obj=train_arr
            )
            save_object(
                file_path=self.data_transformation_config.test_arr_path,
                obj=test_arr
            )

            logging.info("Preprocessing pickle file saved")

            return (
                train_arr,
                test_arr
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
            output_path = home_dir.as_posix() + '/data/processed'
            
            data = load_data(data_path)
            train_data, test_data = split_data(data, params['test_split'], params['seed'])
            save_data(train_data, test_data, output_path)

        except Exception as e:
            raise CustomException(e, sys)

        