# Renewable_Suitability_Predictor

Python Package Index (PyPi): https://pypi.org/project/Renewable-Suitability-Predictor/

This Python package allows the user to predict site suitability for solar or wind energy installations by leveraging machine learning techniques and the NREL Physical Solar Model TMY API (ver 3.2.2)

This package requires that you sign up for an API key at https://developer.nrel.gov/signup/. API rates and limits specified in the NREL documentation apply 

To start, install the module  

```sh 
pip install Renewable-Suitability-Predictor
```

Then import it 
```python
import Renwable-Suitability-Predictor as rsp
```

This module provides the user with two functions 
```python
interface_single_prediction(api_key, email, random_state=-1)

interface_multi_prediction(coordinates_array, model_type, email, api_key, random_state=-1)
```
These functions employ the techniques used in my research paper (https://doi.org/10.58445/rars.682) . They perform 3 main jobs:
1. Gather Typical Meteorological Year (TMY) data from the NREL Physical Solar Model API for each coordinate
2. Transform the time series data to a single dimension
3. Use the Random Forest Classifier (RFC) algorithm to determine suitability

Both functions return a list of two items: 
1. A dataframe with the TMY data, prediction, and probability of the prediction for the coordinate(s) the functions are given 
2. the metrics from training and testing the RFC model on the dataset used in my research project.


```interface_single_prediction``` uses terminal input from the user to predict the suitability of one location.

```interface_multi_prediction``` is for predicting the suitability of many different locations at once. 




## Parameter Details
```coordinates_array```: 2d array-like of shape (n,2). It contains pairs of cooridnates in the order [Latitude, Longitude]. Lat and lon should be given as type **float**.
Example:
```python
coordinates_array = [
    [40.7128, -74.0060],   # New York City, NY
    [34.0549, -118.2426]  # Los Angeles, CA
]
```

```model_type```: type **string**. Either ```'wind'``` or ```'solar'```. Specifies desired prediction type and whether the solar model or wind model should be used.

```api_key```: type **string**. Recieved from the NREL when singing up. Required to interface with the API.

```email```: type **string**. Email you used to recieve your API key. Required to interface with the API.

```random_state```: type **int**. Used as initial seed for dataframe shuffling, train-test shuffling, and RFC. Used for reproducing function outputs. Default -1 means no set random state. If changed by user, ```random_state``` must be in range [0,4294967295]
