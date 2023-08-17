from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = "-e ."
# getting installation files
def get_requirements(file_path:str) -> List[str]:
    '''
    this will return the list of requiremants
    '''
    
    with open(file_path) as file_req:
        requirements = file_req.readlines()
        requirements = [req.replace("\n","") for req in requirements]
        
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements




setup(
    name = 'mlproject_krish',
    version = '1.0.0',
    author = 'ansil',
    author_email = 'ansilproabl@gmail.com',
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt'),
    )
