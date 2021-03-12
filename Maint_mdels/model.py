from main import HomeClass
import pandas as pd

# PRED_MODELS = {}
MAKE_DISTANCE = {}
MAKE_DURATION = {}

def load_service_interval_data():
    data = {}
    # Read data from excel file
    df = pd.read_excel('service_interval.xlsx', header=0)
    df.columns = [['Make','Model','RegYear', 'Months', 'Distance(1000/km)']]
    df = df.drop(index=0)
    
    for index, distance in df.iterrows():
        duration = distance['Months'].values[0]
        make = distance['Make'].values[0]
        distance = distance['Distance(1000/km)'].values[0]
        
        MAKE_DISTANCE[make] = distance
        MAKE_DURATION[make] = duration
        
    data['distance'] = MAKE_DISTANCE
    data['duration'] = MAKE_DURATION
    
    return data

class TimeSeries:
    pass

class Prediction:
    def __init__(self):
        
        self.home_class = HomeClass(data=load_service_interval_data())
        self.home_class.load()
        
    def predict(self, odo_reading, months, make):
        return self.home_class.predict(odo_reading, months, make)
 
    
class MaintModel:
    pass

if __name__ == '__main__':
    predicttion_class = Prediction()
    
    print('---->',predicttion_class.predict(odo_reading=9, months=1, make='BMW'))
    
    