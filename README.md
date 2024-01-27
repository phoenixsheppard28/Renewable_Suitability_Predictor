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

This module provides the user with two methods 
```python
interface_single_prediction(api_key,email,random_state=-1)

interface_multi_prediction(coordinates_array, model_type, email, api_key, random_state=-1)
```




