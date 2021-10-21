from setuptools import setup, find_packages

data = dict(
    name="tfod_utils",
    version="1.0",
    install_requires=[
    ],
    data_files=[],
    packages=find_packages(),
)

if __name__ == '__main__':
    setup(**data)
