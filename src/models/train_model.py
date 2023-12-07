import os
import sys
import numpy as np
import pandas as pd
import pathlib
import yaml

from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor


from src.logger import logging
from src.exception import CustomException
from src.utils import evaluate_model, save_object
from src.features.build_features import DataTransformation


from dataclasses import dataclass
from dvclive import Live
import mlflow

@dataclass
class ModelTrainerConfig:
    
    trained_model_file_path= os.path.join("models", "model2.pkl")
    


class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config= ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr, positive, 
                            alpha, max_iter, l1_ratio, criterion, 
                            n_estimators, learning_rate, subsample,
                            base_estimator, bagging_n_estimators, max_samples,
                            bootstrap, max_features, bootstrap_features, random_state):
        try:
            with mlflow.start_run():
                mlflow.autolog()
                logging.info("split training and testing input data")

                X_train, y_train, X_test, y_test= (
                    train_arr[:,:-1],
                    train_arr[:,-1],
                    test_arr[:, :-1],
                    test_arr[:, -1]
                )
                mlflow.log_param("X_train",X_train)
                mlflow.log_param("y_train", y_train)
                mlflow.log_param("X_test", X_test)
                mlflow.log_param("y_test", y_test)

                models={
                    "LinearRegression":LinearRegression(),
                    "Ridge": Ridge(),
                    "Lasso": Lasso(),
                    "ElasticNet": ElasticNet(),
                    "DecisionTreeRegressor": DecisionTreeRegressor(),
                    "RandomForestRegressor": RandomForestRegressor(),
                    "AdaBoostRegressor": AdaBoostRegressor(),
                    "GradientBoostingRegressor": GradientBoostingRegressor(),
                    "BaggingRegressor": BaggingRegressor()

                }

                param={
                    "LinearRegression":{},
                    "Ridge":{
                        "alpha":alpha,
                        "max_iter": max_iter
                    },
                    "Lasso":{
                        "alpha":alpha,
                        "max_iter": max_iter
                    },
                    "ElasticNet":{
                        "alpha":alpha,
                        "max_iter": max_iter,
                        "l1_ratio":l1_ratio
                    },
                    "DecisionTreeRegressor": {
                        'criterion':criterion,
                        #'splitter':['best','random'],
                        'max_features':max_features,
                    },
                    "RandomForest":{
                        #'criterion':criterion,
                        'max_features':max_features,
                        'n_estimators': n_estimators
                    },
                    "AdaBoostRegressor":{
                        'learning_rate': learning_rate,
                        #'loss':['linear','square','exponential'],
                        'n_estimators': n_estimators
                    },

                    "GradientBoostingRegressor":{
                        #'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                        'learning_rate': learning_rate,
                        'subsample':subsample,
                        #'criterion':['squared_error', 'friedman_mse'],
                        'max_features':max_features,
                        'n_estimators': n_estimators
                    },
                    "BaggingRegressor": {
                        "base_estimator":base_estimator,
                        "n_estimators":bagging_n_estimators,
                        "max_samples":max_samples,
                        "bootstrap":True,
                        "max_features":max_features,
                        "bootstrap_features":True,
                        "random_state": random_state
                        }
                }
                
                
                model_report:dict= evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, params=param)
                print(model_report)
                print("\n=========================\n")
                logging.info(f"model report:{model_report}")

                # To get best model score from dict
                best_model_score= max(sorted(model_report.values()))

                best_model_name= list(model_report.keys())[list(model_report.values()).index(best_model_score)]

                best_model= models[best_model_name]

                
                print(f"Best model found, Model Name:{best_model_name}, R2 Score:{best_model_score}")
                print("\n=========================\n")

                logging.info(f"Best model found, Model Name:{best_model_name}, R2 Score:{best_model_score}")

                ridge= Ridge()
                ridge.fit(X_train,y_train)

                y_pred_test= ridge.predict(X_test)
                print(r2_score(y_test, y_pred_test))

                save_object(
                    file_path=self.model_trainer_config.trained_model_file_path,
                    obj=best_model
                )
                #return r2_square
        
        except Exception as e:
            raise CustomException(e, sys)


    def main():
        try:
            curr_dir = pathlib.Path(__file__)
            home_dir = curr_dir.parent.parent.parent
            params_file = home_dir.as_posix() + '/params.yaml'
            params = yaml.safe_load(open(params_file))["train_model"]

            train_input_file = r'D:\MLOps\MLOps-Projects\New_York_City_Taxi_Trip_Duration\data\interim\train.csv'
            #train_df_data_path = home_dir.as_posix() + train_input_file

            test_input_file = r'D:\MLOps\MLOps-Projects\New_York_City_Taxi_Trip_Duration\data\interim\test.csv'
            #test_df_data_path = home_dir.as_posix() + test_input_file

            DataTransformation_obj= DataTransformation()
            train_arr_data_path, test_arr_data_path= DataTransformation_obj.initiate_data_transformation(train_path=train_input_file, test_path=test_input_file)


            #output_path = home_dir.as_posix() + '/models'
            #pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

            model_train_instance= ModelTrainer()
            model_train_instance.initiate_model_trainer(train_arr=train_arr_data_path, test_arr=test_arr_data_path, positive=params["positive"], 
                                                        alpha=params["alpha"], max_iter=params["max_iter"], l1_ratio=params["l1_ratio"],
                                                        criterion=params["criterion"], n_estimators=params["n_estimators"], learning_rate=params["learning_rate"],
                                                        subsample=params["subsample"], base_estimator=params["base_estimator"], bagging_n_estimators=params["bagging_n_estimators"],
                                                        max_features=params["max_samples"], bootstrap=params["bootstrap"], max_samples=params["max_features"],
                                                        bootstrap_features=params["bootstrap_features"], random_state=params["random_state"]
)

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
        ModelTrainer.main()



