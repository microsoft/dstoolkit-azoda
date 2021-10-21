from setuptools import setup, find_packages

data = dict(
    name="azure_utils",
    version="1.0",
    install_requires=[
        "azureml-sdk>=1.35.0",
    ],
    data_files=[],
    packages=find_packages(),
)

if __name__ == '__main__':
    setup(**data)
