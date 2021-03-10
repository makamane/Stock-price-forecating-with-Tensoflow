
from module import Classification
import time


def trainner():
    print('Loading data...')
    make = {
        'BMW': 10, 'TOYOTA': 15, 
        'RENAULT': 15, 'NISSAN': 15
        }
    
    for mk in make:
        print('%%%%%%%%%%%%%%%%%%%%%%% START {0} %%%%%%%%%%%%%%%%%%'.format(mk))
        object_ = {}
        object_[mk] = make[mk]
        model_class = Classification(make_obt = object_, v_make=mk)
        
        # LableEncoder
        _ = model_class.encoder()
        
        # plitting data
        model_class.split()
        
        # Fitting data
        model_class.fit()
        
          # Model evaluation
        model_class.evaluate(show_obs=False)
        
        # saving model
        model_class.save()
        print('%%%%%%%%%%%%%%%%%%%%%%% END {0} %%%%%%%%%%%%%%%%%%'.format(mk))
        
        # Sleep
        # time.sleep(15)
    return model_class
        
class HomeClass:
    def __init__(self):
        self.maint_model_class = trainner()
        
    def load(self):
        pass
        

if __name__ == "__main__":
    
    home_class = HomeClass()
    