import yaml

with open("params.yaml", "r") as file:
    try:
        print(yaml.safe_load(file))
    except yaml.YAMLError as e:
        print(e)