from setuptools import setup, find_packages

VERSION = '0.2.0'
DESCRIPTION = 'A quick and simple module for predicting a location\'s solar or wind renewable suitability using the NREL PSM API'

# Setting up
setup(
    name="Renewable_Suitability_Predictor",
    version=VERSION,
    author="Phoenix Sheppard",
    author_email="phoenixsheppard28@gmail.com",
    description=DESCRIPTION,
    packages=find_packages(),
    include_package_data=True,
    package_data={'': ['*.csv']},
    install_requires=['pandas', 'numpy', 'requests', 'scikit-learn','setuptools'],
    keywords=['Machine Learning', 'Renewable Energy', 'Solar Energy', 'Wind Energy'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
