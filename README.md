# Renewable_Suitability_Predictor
This Python package allows the user to predict site suitability for solar or wind energy installations by leveraging machine learning techniques and the NREL Physical Solar Model API at 
https://developer.nrel.gov/docs/solar/nsrdb/psm3-2-2-download/ 

This package requires that you sign up for an API key at https://developer.nrel.gov/signup/. API rates and limits specified in the documentation apply 


To start, install the module with 

```sh 
pip install Renewable-Suitability-Predictor
```

Then import it with
```python
import Renwable-Suitability-Predictor as rsp
```

This module provides the user with two functions 
```python
interface_single_prediction(api_key,email,random_state=-1)

interface_multi_prediction(coordinates_array, model_type, email, api_key, random_state=-1)
```
These functions employ the methodologies used in my research paper. They perform 3 main jobs:
1. Gather 2020 meteorological data from the NREL Physical Solar Model API for each coordinate
2. They use the Typical Meteorological Year (TMY) to transform the time series data
3. Use the Random Forest Classifier (RFC) algorithm to determine suitability

Both functions return a list of two things: 
1. A dataframe with the TMY data, prediction, and probability of the prediction for the coordinates they are given 
2. the metrics from training and testing the RFC model on the dataset used in my research project.


```interface_single_prediction``` uses terminal input from the user to gather the location and 




## Parameter Details
