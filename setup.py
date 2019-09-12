from setuptools import setup, find_packages

setup(
    name='Contract',
    version='1.0.0',
    url='https://github.com/jhconning/renegotiation',
    author='Jonathan Conning',
    author_email='jonathan.conning@gmail.com',
    description='A module with python code to support the solution and visualization of renegotiation-proof contracts. See Basu-Conning(2019)',
    packages=find_packages(),    
    install_requires=['numpy', 'scipy'],
)