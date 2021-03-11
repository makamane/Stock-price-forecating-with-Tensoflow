
from module import Classification
# from sklearn.pipeline import Pipeline

PRED_MODELS = {}


def trainner():
    print('Loading data...')
    make = {        
         'UD TRUCKS':20, 'VOLKSWAGEN':15, 'TOYOTA':15, 'NISSAN':15, 'ISUZU':15, 'BMW':10,
        'FORD':20, 'AUDI':15, 'RENAULT':15, 'VOLKSWAGEN':15, 'HONDA':15, 'FIAT':10, 'HYUNDAI':15,
        'ISUZU':15, 'KIA':10, 'CHEVROLET':20, 'MITSUBISHI':15, 'MERCEDES-BENZ':15, 'OPEL':15,
        'MAZDA':15, 'PORSCHE':15, 'SUZUKI':15, 'TATA':15, 'LAND ROVER':15,' SUBARU':12.5,
        'DATSUN':15, 'COLT':15, 'IVECO': 15, 'MAN': 20, 'M A N': 20, 'DUCATI': 15, 'YAMAHA':6,
        'CMC':15, 'BIG BOY': 3
        }
    
    # SPECIALITY
    # UD TRUCKS
    
    #-------------------------
    # MITSUBISHI: 12 month, # KIA : 6 months, # CHEVROLET: 12 month, # DATSUN: 12 months
    # MERCEDES-BENZ: 12 months, # OPEL : 12 months # MAZDA: 12 month, # SUBARU: 6 months
    # PORSCHE: 12 months # SUZUKI: 12 months # TATA: 12 months, # LAND ROVER: 12 months
    # COLT: 12 months # IVECO: 12 months # M A N: 12 months # DUCATI: 12 months
    # YAMAHA : 6 months # CMC: 12 months, # BIG BOY: 6 months
    
    
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
        
    return model_class
        
class HomeClass:
    def __init__(self):
        self.maint_model_class = trainner()
        
    def load(self):
        pass
    
    def predict(self, v_make, odo_reading):
        global home_class
        
        file_path = "trained_models/"+v_make
        model = self.maint_model_class.load_model(file_path, show_accuacy=False)
        
        return model.predict([[0, odo_reading]])[0]

          
# class TimeSeries:
#     def __init__(self):
#         pass # load in new file.....................................
    
#     def load(self):
#         pass
        
if __name__ == "__main__":
    global home_class
    
    home_class = HomeClass()
    
    print(' ')
    print('222222222222222222final')
    results = home_class.predict(v_make='BMW', odo_reading=10)
    print(results)
    print('222222222222222222final')