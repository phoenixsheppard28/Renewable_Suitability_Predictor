#import packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import pkg_resources

#0.1.8 version which is published


def get_model_predictions(prediction_frame,model_type,random_state):

    # read in data from csv, and prepare for training
    # we drop features that were determined insignificant from the feature selection process.
    if(model_type =='wind'):
        df=pd.read_csv(pkg_resources.resource_stream(__name__, 'wind_dataset.csv'))
        df.drop(["State","Name","GHI"], axis=1, inplace=True) 
        
        params = {
            'n_estimators': 300, 'min_samples_split': 4, 'min_samples_leaf': 2, 'max_leaf_nodes': 27, 'criterion': 'gini',
            'max_depth': 5
        }
    elif(model_type=='solar'):
        df=pd.read_csv(pkg_resources.resource_stream(__name__,'solar_dataset.csv'))
        df.drop(["State","Name","Wind_Speed"], axis=1, inplace=True) 
        
        params = {
            'n_estimators': 900, 'min_samples_split': 3, 'min_samples_leaf': 1, 'max_leaf_nodes': 30, 'max_depth': 5, 'criterion': 'gini'
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
    

    #misc prep
    results_dic = {
    'F1-Score': [],
    }


    #preparing functions
    def prepare_semi_supervised_data(data_frame,count):

        if(model_type == 'wind'):
            final_features = ["Temperature", "Wind_Speed","Dew_Point", "Pressure"]
        elif(model_type == 'solar'):
            final_features = ["Temperature", "GHI", "Dew_Point", "Pressure"]

        features = data_frame[final_features]
        labels = data_frame.iloc[:, -1]

        labeled_indices = (labels == 0) | (labels == 1) #boolean mask 
        unlabeled = labels == -1
        X_labeled = features[labeled_indices]
        y_labeled = labels[labeled_indices] 
        X_unknown = features[unlabeled].iloc[:50*(count+1), :]

        return X_labeled, y_labeled, X_unknown

    
    def add_pseudo_label_proba(df, probs, prob_threshold=0.8): #play aroud with value to not have too many -2's  or maybe it can be a user thing
        i = 0 

        while i < df.shape[0] and len(probs) != 0:
            if df.loc[i, "Suitability"] == -1:
                predicted_class = np.argmax(probs[0])

                if probs[0][predicted_class] >= prob_threshold:
                    df.loc[i, "Suitability"] = predicted_class
                else:
                    df.loc[i, "Suitability"] = -2

                probs = probs[1:]
            i += 1

    



    #main training part
    count=0

    while(count<7):

        model = RandomForestClassifier(**params)

        X_labeled, y_labeled, X_unknown = prepare_semi_supervised_data(df, count)

        if(wants_random_state==True):
            X_train_labeled, X_test, y_train_labeled, y_test = train_test_split(X_labeled, y_labeled, test_size=0.2, shuffle=True,random_state=random_state)
        else:
            X_train_labeled, X_test, y_train_labeled, y_test = train_test_split(X_labeled, y_labeled, test_size=0.2, shuffle=True)


        model.fit(X_train_labeled, y_train_labeled)

        prob_predictions = model.predict_proba(X_test)

        f1 = f1_score(y_test, np.argmax(prob_predictions, axis=1))

        
        results_dic["F1-Score"].append(f1)

        if len(X_unknown) > 0:
            prob_labels = model.predict_proba(X_unknown)
            add_pseudo_label_proba(df, prob_labels)
        
        count+=1

    #setting up data from prediction frame
   
    if(model_type=='wind'):
        new_X = prediction_frame[["Temperature", "Wind_Speed","Dew_Point",  "Pressure"]] #now we are predicting unseen data
    elif(model_type=='solar'):
        new_X = prediction_frame[["Temperature", "GHI","Dew_Point",  "Pressure"]] #now we are predicting unseen data

    # print(new_X)

    #now we will predict the labels and add them back in 

    new_suitability = model.predict(new_X)
    new_proba = model.predict_proba(new_X)

    prediction_frame["Probability"] = np.max(new_proba,axis=1,keepdims=True) 
    prediction_frame["Suitability"] = new_suitability


    # print(new_suitability)
    # print(new_proba)

    return prediction_frame , results_dic
     





    




