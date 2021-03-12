import sklearn, warnings
import pandas as pd
import numpy as np
import os, pickle
from sklearn import preprocessing, tree
from sklearn import model_selection

fit_count = 0

def generate_distance_data(mak, make):
    data = pd.DataFrame({'Make':[], 'Reading':[], 'Target':[]}) 
    
    ybelow, yequal, yabove = [], [], []
    xbelow, xequal, xabove = [], [], []

    X,y = [],[]
    mmke = []

    # not_cut = True
    target = make[mak]
    limit = 1500
    # half_limt = 10
    
    for km in range(0, 3000):
        if km % make[mak] == 0:
            if km == 0:
                target =  km
                xequal.append(km)
                yequal.append(0)
                
            elif len(xequal) < limit:
                for t in range(5):
                    xequal.append(km)
                    yequal.append(1)
                    
        else:
            if km < target:
                if len(xbelow) < limit:
                    xbelow.append(km)
                    ybelow.append(0)

            elif km > target:
                if len(xabove) < limit:
                    xabove.append(km)
                    yabove.append(0)
                    
    for x in xbelow:
        X.append(x )
    for x in xequal:
        X.append(x)
    for x in xabove:
        X.append(x)

    for x in ybelow:
        y.append(x)

    for x in yequal:
        y.append(x)
    for x in yabove:
        y.append(x)

    for x in range(len(y)):
        mmke.append(mak)
        
    print('X:',len(X))
    print('y:',len(y))
    print('mmke:',len(mmke))

    df = pd.DataFrame({'Make': mmke, 'Reading':X, 'Target': y})

    return data.append(df)

def generate_duration_data(mak, make_duration_object):
    data = pd.DataFrame({'Make':[], 'Months':[], 'Target':[]}) 
    ybelow, yabove = [], []
    xbelow, xabove = [], []

    X,y = [],[]
    mmke = []
    
    for km in range(0, 200):
        if km >= make_duration_object[mak]:
            xabove.append(km)
            yabove.append(1)       
        else:
            xbelow.append(km)
            ybelow.append(0)

                    
    for x in xbelow:
        X.append(x )
    for x in xabove:
        X.append(x)

    for x in ybelow:
        y.append(x)
    for x in yabove:
        y.append(x)

    for x in range(len(y)):
        mmke.append(mak)
        
    print('X:',len(X))
    print('y:',len(y))
    print('mmke:',len(mmke))

    df = pd.DataFrame({'Make': mmke, 'Months':X, 'Target': y})

    return data.append(df)
   
    
class Classification:
    def __init__(self, make_obt, v_make, train_model_name):
        print('geneating data....')
        self.train_model_name = train_model_name
        self.v_make = v_make
        self.make_obt = make_obt
        self.new_model_path = "trained_models/" + v_make
        if train_model_name == 'make_by_duration.sav':
            self.features_list = ['Make', 'Months']
            self.data = generate_duration_data(mak=v_make, make_duration_object=make_obt)
            
        elif train_model_name == 'make_by_distance.sav':
            self.features_list = ['Make', 'Reading']
            self.data = generate_distance_data(mak=v_make, make=make_obt)
        print("Data target categories")
        print(self.data.groupby('Target').size())
        
    def encoder(self):
        print('Encoding make...')
        self.LabelEncoder = preprocessing.LabelEncoder()
        self.LabelEncoder.fit(self.data['Make'])
        self.LabelEncoder.classes_
        # le.transform(data['Make'])
        self.data['Make'] = self.LabelEncoder.transform(self.data['Make'])
        return self.LabelEncoder
        
    def split(self):
        print('Splitting data...')
        self.X = np.array(self.data[self.features_list])
        self.y = np.array(self.data[['Target']])
        
        # K-fold split
        kf = model_selection.KFold(n_splits=int(self.make_obt[self.v_make]), shuffle=False)
        kf.get_n_splits(self.X)
        # print(kf)
        for train_index, test_index in kf.split(self.X):
        #     print("TRAIN:", train_index, "TEST:", test_index)
            self.X_train, self.X_test = self.X[train_index], self.X[test_index]
            self.y_train, self.y_test = self.y[train_index], self.y[test_index]
                
    def fit(self):
        print('Fitting data...')
        self.model = tree.DecisionTreeClassifier()
        self.model = self.model.fit(self.X_train, self.y_train)
        print('pre-evaluateing...')
        y_pred = self.model.predict(self.X_test)
        acc = sklearn.metrics.accuracy_score(self.y_test, y_pred)
        print('Accuracy:', acc)
        print('------------------------')
        # if acc < 90:
        #     print('------------------------>>>>')
        #     # for pred, act, tt in zip(y_pred, self.y_test, self.X_test):
        #     #     print(tt[1], '--> predicted: {0}| Actual: {1}'.format(pred, act[0]))
            
            
        #     print('------------------------>>>>')
        
    def evaluate(self, show_obs=False, show_warning=True):
        print('Model evaluation...')
        self.predicted = []
        self.actual = []
        
        if self.train_model_name == 'make_by_distance.sav':
            test_events = 200
            for mk in self.make_obt:
                print('       ',mk, ' | ',  self.make_obt[mk])
                print('-------------------')
                target = self.make_obt[mk]
                brend_code = self.LabelEncoder.transform([mk])[0]
                for km in range(test_events):
                    res = self.model.predict([[brend_code, km]])[0]
                    self.predicted.append(res)
                    if km % target == 0:
                        if km > 0:
                            self.actual.append(1)
                            if show_obs:
                                print(km, 'Predicted: {0}| Actual: {1}'.format(res, 1))
                        else:
                            self.actual.append(0)
                            if show_obs:
                                print(km, 'Predicted: {0}| Actual: {1}'.format(res, 0))
                    else:
                        self.actual.append(0)
                        if show_obs:
                            print(km,'Predicted: {0}| Actual: {1}'.format(res, 0))
                                        
        if self.train_model_name == 'make_by_duration.sav':
            test_events = 1000
            
            for mk in self.make_obt:
                print(mk, '_____',  self.make_obt[mk])
                target = self.make_obt[mk]
                brend_code = 0
                for km in range(1000):
                    res = self.model.predict([[brend_code, km]])[0]
                    self.predicted.append(res)
                    if km >= target:
                        self.actual.append(1)
                        # print(km, 'predicted: {0}| Actual: {1}'.format(res, 1))
                    else:
                        self.actual.append(0)
            #             print(km,'predicted: {0}| Actual: {1}'.format(res, 0))
            
            
            
        if self.v_make in ['BIG BOY']:
            self.score = sklearn.metrics.accuracy_score(self.actual, self.predicted)
        else:
            self.score = sklearn.metrics.accuracy_score(self.actual, self.predicted)
               
        print("Accuracy based on 1k events: {0}%".format(np.round(self.score*100, 2)))
        if (self.score < 0.83)and(show_warning):
            warnings.warn('Accuracy score of {0}%'.format(np.round(self.score*100, 2)))

    def save(self):
        global fit_count
        
        if self.score >= 0.83:
            print('Saving fitted model...', end="\r")
            # saving the label_ids in to pickle
            file = self.new_model_path+'/'+self.train_model_name
            
            if not os.path.exists(self.new_model_path):
                os.makedirs(self.new_model_path)
                
            with  open(file, 'wb') as f:
                pickle.dump(self.model, f)
            print('Saved fitted model...', end="\r")
        else:
  
            while self.score < 0.83:
                Classification.split(self)
                Classification.fit(self)
                Classification.evaluate(self, show_warning=False)
                fit_count +=1
                if fit_count>200:
                    break
            if (self.score < 0.83) and (fit_count<=200):
                Classification.save(self)
            print('!!!!!!!!!!!!!!!!!!!!!')
            warnings.warn('Saving failed due to low accuracy')
            
    def load_model(self, model_path, show_accuacy=False):
        # saving the label_ids in to pickle
        file = str(model_path)+'/'+self.train_model_name
        # with  open(file, 'rb') as f:
        self.loaded_model = pickle.load(open(file, 'rb'))
            
        if show_accuacy:
            Classification.evaluate()
        return self.loaded_model
        