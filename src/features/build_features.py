import pandas as pd
import numpy as np
import os
import sys
import pathlib
import yaml

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
    train_arr_path= os.path.join("data\processed", "train_arr.npz")
    test_arr_path= os.path.join("data\processed", "test_arr.npz")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config= DataTransformationConfig()


    def get_data_transformer_obj(self,train_df, test_df):
        '''
        This function is responsible for data transformation
        '''

        try:
            Num_columns=train_df.columns[(train_df.dtypes == float) | (train_df.dtypes == int)]
            Cate_columns=train_df.columns[train_df.dtypes=="object"]

            store_and_fwd_flag=train_df[["store_and_fwd_flag"]].columns

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
                    #("SS", trf2, Num_columns)
                ], remainder="passthrough")
            
            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        
    
    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df= pd.read_csv(test_path)

            logging.info("Read train and test data")

            logging.info(f"train DataFrame's head:\n {train_df.head().to_string()}")
            logging.info(f"test DataFrame's head:\n {test_df.head().to_string()}")

            preprocessing_obj= self.get_data_transformer_obj(train_df=train_df, test_df=test_df)

            target_col_name=['trip_duration']
            
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

            logging.info("Preprocessing pickle file saved")

            np.savez(self.data_transformation_config.train_arr_path, train_arr)
            np.savez(self.data_transformation_config.test_arr_path, test_arr)

            print(train_arr[:, -1])
            return(
                train_arr,
                test_arr
            )


        except Exception as e:
            raise CustomException(e, sys)
        
    def main():
        try:
            curr_dir = pathlib.Path(__file__)
            home_dir = curr_dir.parent.parent.parent
            '''params_file = home_dir.as_posix() + '/params.yaml'
            params = yaml.safe_load(open(params_file))["make_dataset"]'''

            input_file_train = sys.argv[1]
            train_path = home_dir.as_posix() + input_file_train

            input_file_test = sys.argv[2]
            test_path= home_dir.as_posix() + input_file_test

            Data_transformation_instance=DataTransformation()
            Data_transformation_instance.initiate_data_transformation(train_path=train_path, test_path=test_path)
            

        except Exception as e:
            raise CustomException(e, sys)

if __name__=="__main__":
    DataTransformation.main()
