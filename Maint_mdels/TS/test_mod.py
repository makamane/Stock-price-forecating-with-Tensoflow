import math
import pandas as pd
from t_series import TimeSeries_analysis

# Load data
LOADED_DATA = pd.read_csv('testdata.csv')
# convet data to ts data
TIME_SERIES = TimeSeries_analysis(data=LOADED_DATA, index_tearget='Month', target_values='#Passengers', window=12)

def test_sqrt():
   num = 25
   assert math.sqrt(num) == 5
   
def test_split_data():
    
    train_size = int(len(LOADED_DATA)*.80)
    test_size = int(len(LOADED_DATA) - train_size)
    assert train_size == TIME_SERIES.split_data()[0]
    assert test_size == TIME_SERIES.split_data()[1]
   
   
   


# if __name__ == "__main__":
#     print('222222',len(LOADED_DATA))
    
 

