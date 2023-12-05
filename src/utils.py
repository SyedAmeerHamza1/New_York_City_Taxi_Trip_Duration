import os
import sys
import pandas as pd
import numpy as np
import pickle
import pathlib

from src.exception import CustomException

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    try:
        dir_path= os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    
def save_obj(train, test, preprocessed, output_path):
        # Save the split datasets to the specified output path
        pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
        train.to_csv(output_path + '/train.csv', index=False)
        test.to_csv(output_path + '/test.csv', index=False)
        preprocessed.to_p(output_path + '/test.csv', index=False)
    

def evaluate_model(X_train, y_train, X_test, y_test, models, params):
    try:
        report={}
        for i in range(len(list(models))):
            model= list(models.values())[i]
            param= params[list(models.keys())[i]]

            gs= GridSearchCV(model,param, cv=3)

            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params)
            model.fit(X_train, y_train)

            y_train_pred= model.predict(X_train)
            y_test_pred= model.predict(X_test)

            train_model_score= r2_score(y_train, y_test_pred)
            test_model_score= r2_score(y_test, y_test_pred)

            report[list(model.keys())[i]]=test_model_score

            return report
        
    except Exception as e:
        raise CustomException(e, sys) 