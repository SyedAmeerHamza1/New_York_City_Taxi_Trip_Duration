import os
import sys
import numpy as np
import pandas as pd
import pathlib

from dataclasses import dataclass

from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor


from src.logger import logging
from src.exception import CustomException
from src.utils import evaluate_model, save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path= os.path.join("models", "model.pkl")

class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config= ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("split training and testing input data")

            X_train, y_train, X_test, y_test= (
                train_arr[:,: -1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )

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

            params={
                "Linear_Regressen":{
                    #"positive":["True", "False"]
                },
                "Ridge":{
                    "alpha":[1.0,0.5,1.5,0.8],
                    "max_iter": [1000,800,900,1500]
                },
                "Lasso":{
                    "alpha":[1.0,0.5,1.5,0.8],
                    "max_iter": [1000,800,900,1500]
                },
                "Elastic_Net":{
                    "alpha":[1.0,0.5,1.5,0.8],
                    "max_iter": [1000,800,900,1500],
                    "l1_ratio":[0.5, 0.8,0.2,0.7]
                },
                "Decision_Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random_Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "AdaBoost_Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                },

                 "Gradient_Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "BaggingRegressor": {
                    "base_estimator":[LinearRegression(), Ridge(), Lasso(), DecisionTreeRegressor()],
                    "n_estimators":[500, 100, 250],
                    "max_samples":[0.25, 0.50],
                    "bootstrap":[True],
                    "max_features":[0.5, 0.25],
                    "bootstrap_features":[True],
                    "random_state":[42]}
            }

            model_report:dict= evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, params=params)

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

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
        
        except Exception as e:
            raise CustomException(e, sys)
        

    def main():
        try:
            curr_dir = pathlib.Path(__file__)
            home_dir = curr_dir.parent.parent.parent
            params_file = home_dir.as_posix() + '/params.yaml'
            params = yaml.safe_load(open(params_file))["train_model"]

            input_file = sys.argv[1]
            data_path = home_dir.as_posix() + input_file
            output_path = home_dir.as_posix() + '/models'
            pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
            
            TARGET = 'Class'
            train_features = pd.read_csv(data_path + '/train.csv')
            X = train_features.drop(TARGET, axis=1)
            y = train_features[TARGET]

            trained_model = train_model(X, y, params['n_estimators'], params['max_depth'], params['seed'])
            save_model(trained_model, output_path)

        except Exception as e:
            raise CustomException(e, sys)

    

if __name__ == "__main__":
    ModelTrainer.main()

