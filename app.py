import mlflow

def calculate(x,y): 
    
    return (x-y)


if __name__=="__main__":
    with mlflow.start_run():
        x, y=5000, 100
        result=calculate(x,y)
        mlflow.log_param("x",x)
        mlflow.log_param("y",y)
        mlflow.log_param("result", result)