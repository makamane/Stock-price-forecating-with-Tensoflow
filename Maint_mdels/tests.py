

import math
import os, pickle

def test_main_models():
    make_obt = {'UD TRUCKS':20, 'VOLKSWAGEN':15, 'TOYOTA':15, 'NISSAN':15, 'ISUZU':15, 'BMW':10,
        'FORD':20, 'AUDI':15, 'RENAULT':15, 'VOLKSWAGEN':15, 'HONDA':15, 'FIAT':10, 'HYUNDAI':15,
        'ISUZU':15, 'KIA':10, 'CHEVROLET':20, 'MITSUBISHI':15, 'MERCEDES-BENZ':15, 'OPEL':15,
        'MAZDA':15, 'PORSCHE':15, 'SUZUKI':15, 'TATA':15, 'LAND ROVER':15,' SUBARU':12.5,
        'DATSUN':15, 'COLT':15, 'IVECO': 15, 'MAN': 20, 'M A N': 20, 'DUCATI': 15, 'YAMAHA':6,
        'CMC':15, 'BIG BOY': 3
        }
    
    for mk in make_obt:
        # load models
        file = "trained_models/" +mk+'/trainner.sav'
        loaded_model = pickle.load(open(file, 'rb'))
        
        print('       ',mk, ' | ',  make_obt[mk])
        print('-------------------')
        # service interval
        target = make_obt[mk]
        brend_code = 0 # make encoded int
        
        for km in range(500):
            results = loaded_model.predict([[brend_code, km]])[0]
            if km % target == 0:
                if km > 0:
                    assert int(results) == 1
                else:
                    assert int(results) == 0
            else:
                assert int(results) == 0

