from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT="-e ."

def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements



'''setup(
    name='src',
    packages=find_packages(),
    #install_requires=get_requirements("requirements.txt"),
    version='0.1.0',
    description='Building a model that predicts the total ride duration of taxi trips in New York City.',
    author='Syed Ameer Hamza',
    license='MIT'
    
)'''


setup(
    name='New_York_City_Taxi_Trip_Duration',
    version='0.0.1',
    author='Syed Ameer Hamza',
    author_email='Syedsoahil116@gamil.com',
    install_requires=get_requirements("requirements.txt"),
    packages=find_packages()
)
