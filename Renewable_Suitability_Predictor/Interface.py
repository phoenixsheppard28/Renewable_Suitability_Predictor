import requests
import pandas as pd
import numpy as np
import io
from .backend_methods import get_model_predictions

#need to make api key and email parameters at end

#dont implement semi supervised 


def make_wkt(lat, lon):
        wkt_point = f"POINT({lon} {lat})"
        return wkt_point

def interface_multi_prediction(coordinates_array, model_type, email, api_key, random_state=-1):

    
    mode_wind = False
    mode_solar= False
    pramaters={
        "api_key":api_key,  #their email and API key will go here 
        "email":email,
        "names":"tmy-2022"  # year for which data is extracted from, do not change
    }

    #making sure parameters are valid 
    if(model_type == 'solar'):
        mode_solar=True
    elif(model_type == 'wind'):
        mode_wind=True
    else: 
        raise Exception("Invalid mode type, please pass either 'solar' or 'wind'")
    
    if(np.shape(coordinates_array)[1]!=2):
        raise Exception("Invalid coordinates array, please pass a 2d array of shape n,2")
    
    

    #transforming= averaging part? 
    #maybe make 1 df and concatenate them on top of each other

    AVG_dic={ 
        "Temperature": [],
        "Pressure":[],
        "Dew_Point":[],
        }
    val_list=[]

    if(mode_wind==True):
        AVG_dic["Wind_Speed"]=[]

        val_list.append("DHI Units")
        val_list.append("Dew Point Units")
        val_list.append("Latitude") #dew point
        val_list.append("GHI Units")

    elif(mode_solar==True):
        AVG_dic["GHI"] = []

        val_list.append("DHI Units")
        val_list.append("Dew Point Units")
        val_list.append("Latitude")  #dew point
        val_list.append("Elevation")


    # print(AVG_dic["Temperature"])
    # print(val_list)
    # print(df.head(5))
    # print(pd.to_numeric(df[val_list[0]]).mean())

    

    
    for pair in coordinates_array:
        if ( (isinstance(pair[0],float)) & (isinstance(pair[1],float)) ):
            temp_wkt= (make_wkt(pair[0],pair[1]))  # adding them to a list of WKT to check for 

            print(f"Starting request for {pair} ...")

            r=requests.get(f"http://developer.nrel.gov/api/nsrdb/v2/solar/psm3-2-2-tmy-download.csv?wkt={temp_wkt}",params=pramaters) #old api is /api/nsrdb/v2/solar/psm3-download.csv
            print("Finished!")
            df=pd.read_csv(io.StringIO(r.content.decode('utf-8'))) 
            df=df.iloc[2:]

            
            if(df.columns.values.tolist()[8]=="Data processing failure.]}"):  #or need to find a different status code that means its bad becasue 400 is all encompassing
                raise Exception(f"Coordinate pair '{pair}' at index {coordinates_array.index(pair)} is not within the range of the api")
            elif(r.status_code==400):
                raise Exception("There is a problem with the API or your key. Please check the API and your key's activation status.  Please also check that your email and api key parameters are correct")
            elif(r.status_code==200):
                count=0
                for i in AVG_dic.keys():
                    AVG_dic[i].append(pd.to_numeric(df[val_list[count]]).mean()) 
                    count+=1

            else:
                print(f"other error, status code: {r.status_code}")
        
        else:    
            raise Exception("Invalid coordinates array, each pair of coordinates must be given as floats")

    coordinates_array = np.array(coordinates_array)

    true_frame=pd.DataFrame(AVG_dic)
    true_frame.insert(0,"Latitude",coordinates_array[:,0])
    true_frame.insert(1,"Longitude",coordinates_array[:,1])
    true_frame.insert(6,"Probability",None) #will be filled in by the model 
    true_frame.insert(7,"Suitability",-1)  #no brackets for both becasue it will fill in the whole column

    

    return get_model_predictions(true_frame,model_type,random_state)




    








    

#maybe make a method overload version or something to check if parameters are passed
def interface_single_prediction(email,api_key,random_state=-1):
    pramaters={
        "api_key":api_key,  #their email and API key will go here 
        "email":email,
        "names":"tmy-2022"  # year for which data is extracted from, do not change
    }



    #starter stuff

    mode_wind = False
    mode_solar= False
    print("Hello, this program will allow you to easily predict the suitability for a given coordiante")
    while(True):
        mode_input= str(input("Please enter your desired prediction type: 'wind' or 'solar' :"))

        if(mode_input=='wind'):
            mode_wind = True
            break
        elif(mode_input=='solar'):
            mode_solar = True
            break
        print("Please enter a valid prediction type")


    # coordinate part

    while(True):
        user_coordinates = list(map(float, input("Enter the Latitude and Longitude of the coordinate you wish to investigate (seperate each with a space): ").split()))
        print(user_coordinates[0], user_coordinates[1])

        coordinate= make_wkt(user_coordinates[0],user_coordinates[1])
        print("Starting Request...")
        r=requests.get(f"http://developer.nrel.gov/api/nsrdb/v2/solar/psm3-2-2-tmy-download.csv?wkt={coordinate}",params=pramaters) #old api is /api/nsrdb/v2/solar/psm3-download.csv

        print(f"Status code: {r.status_code}")

        df=pd.read_csv(io.StringIO(r.content.decode('utf-8'))) #temp addition
        if(df.columns.values.tolist()[8]=="Data processing failure.]}"):
            print("Please enter a valid pair of coordinates within the contiguous United States")
            continue
        elif(r.status_code==400):
            print("There is a problem with the API or your key. Please check the API and your key's activation status. Please also check that your email and api key parameters are correct")
            continue
        break

    print("Request Finished!")
    df=pd.read_csv(io.StringIO(r.content.decode('utf-8')))

    


    #transforming/ averaging part
    
    AVG_dic={ 
        "Temperature": [],
        "Pressure":[],
        "Dew_Point":[],
        }
    val_list=[]

    if(mode_wind==True):
        AVG_dic["Wind_Speed"]=[]

        val_list.append("DHI Units")
        val_list.append("Dew Point Units")
        val_list.append("Latitude") #dew point
        val_list.append("GHI Units")

    elif(mode_solar==True):
        AVG_dic["GHI"] = []

        val_list.append("DHI Units")
        val_list.append("Dew Point Units")
        val_list.append("Latitude")  #dew point
        val_list.append("Elevation")

    df=df.iloc[2:]

    

    count=0
    for i in AVG_dic.keys():
        AVG_dic[i].append(pd.to_numeric(df[val_list[count]]).mean())
        count+=1


    true_frame=pd.DataFrame(AVG_dic)
    true_frame.insert(0,"Latitude",[user_coordinates[0]])
    true_frame.insert(1,"Longitude",[user_coordinates[1]])
    true_frame.insert(6,"Probability",None) #will be filled in by the model 
    true_frame.insert(7,"Suitability",-1)  #-1 means the suitability is unpredicted 


    return get_model_predictions(true_frame,mode_input,random_state)

    #returns the dataframe with predicted suitability label and probability added, along with training metrics for the model



