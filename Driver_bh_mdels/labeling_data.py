'''
This needs to be tested, by for now the assumtion is that if the
acceleration of the y is half or more than of the x then is harsh event
'''
import pandas as pd


# Load data
LOADED_DATA = pd.read_csv('testdata.csv', delimiter=',')

class LabelingTheData:
    def harsh_con(x, y):
        if (x*10>=1):
            if abs(y) >= abs(x)*0.7:
                event = 'Harsh'
            else:
                event = 'NotHarsh'
        else:
            event = 'NotHarsh'
        return event
    
    def ha(self):
        pass
    
    def hb(self):
        pass
    
class AxisDetection:
    def __init__(self, data):
        self.data = data
        print('AxisDetection started...')
        print('Date detected is with {0} rows'.format(len(data)))
        ''' This will only work while the acceleration > 1:'''
        if len(data) > 2:
            data_colums = data.columns
            print(data_colums)
            # z_axis = data['Z']
            
            
        
    



# HA = 1.28220 m/s2
# HB = −1.3230 m/s2

if __name__ == "__main__":

    print(LOADED_DATA.head(5))
    axis_detec = AxisDetection(data=LOADED_DATA)
    
