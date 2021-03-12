import pickle
from module import Classification


def make_trainner(make_distance_train=True):
    print('Loading data...')
    
    if make_distance_train:
        with  open('make_by_distance.pickle', 'rb') as file:
            make = pickle.load(file)
        model_saves_by = 'make_by_distance.sav'
        
    else:
        with  open('make_by_duration.pickle', 'rb') as file:
            make = pickle.load(file)
            model_saves_by = 'make_by_duration.sav'
        
    for mk in make:
        print('%%%%%%%%%%%%%%%%%%%%%%% START {0} %%%%%%%%%%%%%%%%%%'.format(mk))
        object_ = {}
        object_[mk] = make[mk]
        model_class = Classification(make_obt = object_, v_make=mk, train_model_name=model_saves_by)
        
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
    def __init__(self, data):
        self.v_make_object = data['distance']
        
        # saving the make distance data in to pickle
        with  open("make_by_distance.pickle", 'wb') as f:
            pickle.dump(data['distance'], f)
            
        # saving the make duration data in to pickle
        with  open("make_by_duration.pickle", 'wb') as f:
            pickle.dump(data['duration'], f)
            
        self.maint_distance_model_class = make_trainner()
        print('---------------------------------')
        print("   /|\\")
        print("  / | \\")
        print(" /  |  \\")
        print("/___|___\\")
        print("\   |   /")
        print(" \  |  /")
        print("  \ | /")
        print("   \|/")
        print('---------------------------------')
        self.maint_duration_model_class = make_trainner(make_distance_train=False)
        
        
        
    def load(self):
        print('Loading models...')
        self.loaded_models_object = {}
        
        for make in self.v_make_object:
            file_path1 = "trained_models/"+ make +"/make_by_distance.sav"
            file_path2 = "trained_models/"+ make +"/make_by_duration.sav"
            
            self.loaded_models_object[make] = {
                'distance': pickle.load(open(file_path1, 'rb')),
                'duration': pickle.load(open(file_path2, 'rb'))
                }
        print("|\\        /|    -  _    |-      _____ |")
        print("| \\      / |   -    -   |  -   |      |")
        print("|  \\    /  |  -      -  |   -  |      |")
        print("|   \\  /   | -        - |    - |      |")
        print("|    \\/    |  -      -  |    - |----- |")
        print("|          |   -    -   |   -  |      |")
        print("|          |    -  -    |  -   |      |")
        print("|          |     -      |-     |_____ |_____s Ready...")
        
    def predict(self, odo_reading, months, make):
        model = self.loaded_models_object[make]
        dis_model_results = int(model['distance'].predict([[0, odo_reading]])[0])
        dur_model_results = int(model['duration'].predict([[0, months]])[0])
        
        if (dis_model_results==0) and (dur_model_results==0):
            return 0
        else:
            return 1
          