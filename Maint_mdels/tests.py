
import pickle

def test_main_distance_models():
    
    with  open('make_by_distance.pickle', 'rb') as file:
        make_obt = pickle.load(file)

    for mk in make_obt:
        # load models
        file = "trained_models/" +mk+'/make_by_distance.sav'
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
                
def test_main_duration_models():
    
    with  open('make_by_duration.pickle', 'rb') as file:
        make_dur_obt = pickle.load(file)

    for mk in make_dur_obt:
        # load models
        file = "trained_models/" +mk+'/make_by_duration.sav'
        loaded_model = pickle.load(open(file, 'rb'))
        print('       ',mk, ' | ',  make_dur_obt[mk])
        print('-------------------')
        # service interval
        target = make_dur_obt[mk]
        brend_code = 0 # make encoded int
        
        for km in range(200):
            results = loaded_model.predict([[brend_code, km]])[0]
            if km >= target:
                assert int(results) == 1

            else:
                assert int(results) == 0

