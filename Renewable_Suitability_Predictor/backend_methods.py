#import packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import hamming_loss,f1_score




def get_wind_model_predictions(prediction_frame,random_state):
    # read in data from csv, and prepare for training
    # we drop features that were determined insignificant from the feature selection process.
    df=pd.read_csv('wind_dataset.csv')
    df.drop(["State","Name","Surface_Albedo","Precipitable_Water","GHI"], axis=1, inplace=True) 
     
    params = {
        'n_estimators': 300, 'min_samples_split': 4, 'min_samples_leaf': 2, 'max_leaf_nodes': 27, 'criterion': 'gini',
        'max_depth': 5
    }

    #random_state checking 
    wants_random_state = False

    if(isinstance(random_state,int) & ((random_state <= 4294967295 & random_state >= 0) | (random_state ==-1))):
        if(random_state == -1):
            df=df.sample(frac=1)  

        else:
            params['random_state'] = random_state
            df=df.sample(frac=1,random_state=random_state) 
            wants_random_state=True

    else:
        raise Exception("Invalid random state, please pass an integer in range [0, 4294967295], or -1 for no random state")
    



    #preparing functions
    def prepare_supervised_data(data_frame):
        final_features = ["Temperature", "Wind_Speed", "Relative_Humidity", "Pressure"]
        features = data_frame[final_features]
        labels = data_frame.iloc[:, -1]

        labeled_indices = labels != -1
        X_labeled = features[labeled_indices]
        y_labeled = labels[labeled_indices]

        return X_labeled, y_labeled

    



    #main training part

    model = RandomForestClassifier(**params)
    X_labeled, y_labeled = prepare_supervised_data(df)
    if(wants_random_state==True):
        X_train_labeled, X_test, y_train_labeled, y_test = train_test_split(X_labeled, y_labeled, test_size=0.2, shuffle=True,random_state=random_state)
    else:
        X_train_labeled, X_test, y_train_labeled, y_test = train_test_split(X_labeled, y_labeled, test_size=0.2, shuffle=True)


    model.fit(X_train_labeled, y_train_labeled)

    prob_predictions = model.predict_proba(X_test)

    accuracy = model.score(X_test, y_test)
    f1 = f1_score(y_test, np.argmax(prob_predictions, axis=1))
    hamming_loss_value = hamming_loss(y_test, np.argmax(prob_predictions, axis=1))

    metrics = {
        'Accuracy': accuracy,
        'F1-Score': f1,
        'Hamming Loss': hamming_loss_value
    }

    #setting up data from prediction frame
   

    new_X = prediction_frame[["Temperature", "Wind_Speed", "Relative_Humidity", "Pressure"]] #now we are predicting unseen data

    # print(new_X)

    #now we will predict the labels and add them back in 

    new_suitability = model.predict(new_X)
    new_proba = model.predict_proba(new_X)

    prediction_frame["Probability"] = np.max(new_proba,axis=1,keepdims=True) 
    prediction_frame["Suitability"] = new_suitability


    # print(new_suitability)
    # print(new_proba)

    return prediction_frame , metrics
     




def get_solar_model_predictions(prediction_frame,random_state):
   # read in data from csv, and prepare for training
    # we drop features that were determined insignificant from the feature selection process.
    df=pd.read_csv('solar_dataset.csv')
    df.drop(["State","Name","Surface_Albedo","Precipitable_Water","Wind_Speed"], axis=1, inplace=True) 
     
    params = {
        'n_estimators': 300, 'min_samples_split': 4, 'min_samples_leaf': 2, 'max_leaf_nodes': 27, 'criterion': 'gini',
        'max_depth': 5
    }

    #random_state checking 
    wants_random_state = False

    if(isinstance(random_state,int) & ((random_state <= 4294967295 & random_state >= 0) | (random_state ==-1))):
        if(random_state == -1):
            df=df.sample(frac=1)  

        else:
            params['random_state'] = random_state
            df=df.sample(frac=1,random_state=random_state) 
            wants_random_state=True

    else:
        raise Exception("Invalid random state, please pass an integer in range [0, 4294967295], or -1 for no random state")
    



    #preparing functions
    def prepare_supervised_data(data_frame):
        final_features = ["Temperature", "GHI", "Relative_Humidity", "Pressure"]
        features = data_frame[final_features]
        labels = data_frame.iloc[:, -1]

        labeled_indices = labels != -1
        X_labeled = features[labeled_indices]
        y_labeled = labels[labeled_indices]

        return X_labeled, y_labeled

    



    #main training part

    model = RandomForestClassifier(**params)
    X_labeled, y_labeled = prepare_supervised_data(df)
    if(wants_random_state==True):
        X_train_labeled, X_test, y_train_labeled, y_test = train_test_split(X_labeled, y_labeled, test_size=0.2, shuffle=True,random_state=random_state)
    else:
        X_train_labeled, X_test, y_train_labeled, y_test = train_test_split(X_labeled, y_labeled, test_size=0.2, shuffle=True)


    model.fit(X_train_labeled, y_train_labeled)

    prob_predictions = model.predict_proba(X_test)

    accuracy = model.score(X_test, y_test)
    f1 = f1_score(y_test, np.argmax(prob_predictions, axis=1))
    hamming_loss_value = hamming_loss(y_test, np.argmax(prob_predictions, axis=1))

    metrics = {
        'Accuracy': accuracy,
        'F1-Score': f1,
        'Hamming Loss': hamming_loss_value
    }
    
    #setting up data from prediction frame
   

    new_X = prediction_frame[["Temperature", "GHI", "Relative_Humidity", "Pressure"]] #now we are predicting unseen data

    # print(new_X)

    #now we will predict the labels and add them back in 

    new_suitability = model.predict(new_X)
    new_proba = model.predict_proba(new_X)

    prediction_frame["Probability"] = np.max(new_proba,axis=1,keepdims=True) 
    prediction_frame["Suitability"] = new_suitability


    # print(new_suitability)
    # print(new_proba)

    return prediction_frame , metrics
     


    



    

