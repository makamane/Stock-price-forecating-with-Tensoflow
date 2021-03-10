import pandas as pd
import numpy as np
import matplotlib.pylab as plt
# %matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6
import os

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from statsmodels.tsa.arima.model import ARIMA

from statsmodels.tsa.stattools import acf, pacf 

import pmdarima as pm
from sklearn.metrics import r2_score
import pickle
from sklearn import metrics



def model_test(expected, predicted):
    
    y_pred = []
    y_true = []
    forecast_errors = []
    forecast_errors_abs = []

    for pred, actu in zip(predicted, expected):
        y_pred.append(pred)
        y_true.append(actu)
        
        forecast_errors.append(actu-pred)
        forecast_errors_abs.append(abs(actu-pred))
    
    # Zero indicate no error
    # Mean Absolute Error
    MAE = metrics.mean_absolute_error(expected, predicted)
    print('MAE:', MAE)
    
    # Mean Squared Error
    # forecast_errors_sqr = [val**2 for val in forecast_errors]
    mse = metrics.mean_squared_error(expected, predicted)
    print('MSE:', mse)
    
    # Root Mean Squared Error
    rmse = np.sqrt(mse)
    print('RMSE:',rmse)
    
def get_first_item(list_var):
    try:
        list_var = list_var[0][0]
    except Exception as err:
        if str(err)[:32] == 'invalid index to scalar variable':
            list_var = list_var[0]
        else:
            raise Exception(err)
    return list_var
    
class TimeSeries_analysis:
    def __init__(self, data, index_tearget, target_values, window):
        self.data = data
        self.window = window
        self.target_values = target_values
        print(self.data.head())
        print('\n Data Types:')
        print(self.data.dtypes)
        print('-------------------------')
        
        # con = self.data[index_tearget]
        self.data[index_tearget] = pd.to_datetime(self.data[index_tearget])
        self.data.set_index(index_tearget, inplace=True)
        #check datatype of index
        # print('Data Index:',"\n", self.data.index)
        print('-------------------------')
        
        #convert to time series:
        self.ts = data[target_values]
        
        self.ts_log = np.log(self.ts)
        
        print('Time Series first 10:', self.ts.head(10))
        print('-------------------------')
        
    def params_estimation(self):
        model = pm.auto_arima(self.data[self.target_values], d=1, D=1,
                              m=self.window, trend='c', seasonal=True,
                              start_p=0, start_q=0, max_order=6, test='adf',
                              stepwise=True, trace=True)
        prms = str(model).split('[')[0]
        prms = prms.replace('ARIMA', '')
        prms = prms.replace(')(', ';')
        prms = prms.replace('(', '')
        prms = prms.replace(')', '')
        pdq = prms.split(';')[0]
        PDQ = prms.split(';')[1]  
        
        self.p,self.d,self.q = int(pdq[1]), int(pdq[3]), int(pdq[5])
        self.P,self.D,self.Q= int(PDQ[0]), int(PDQ[2]), int(PDQ[4])

        return (self.p, self.d, self.q, self.P, self.D, self.Q)
            
    def data_plt(self, show_ts=False):
        print('Is it clear from the plot that there is an overall increase in the trend and with some seasonality in it?')
        if show_ts:
            plt.plot(self.ts)
            plt.show()
        
    def stationarity_testing(self, show_p_value=False, show_rolmean_rolstd=False, data=None):
        print('++++++++++++++++++++++++',data)
        if data is not None:
            print('Data is given....')
            timeseries = data
        else:
            print('Data not given....')
            timeseries = self.ts
        
        rolmean = timeseries.rolling(self.window).mean()
        rolstd = timeseries.rolling(self.window).std()
    
        #Plot rolling statistics:
        if show_rolmean_rolstd:
            orig = plt.plot(timeseries, color='blue',label='Original')
            mean = plt.plot(rolmean, color='red', label='Rolling Mean')
            std = plt.plot(rolstd, color='black', label = 'Rolling Std')
            
            plt.legend(loc='best')
            plt.title('Rolling Mean & Standard Deviation')
            plt.show()
    
        
        #Perform Dickey-Fuller test:
        print('Results of Dickey-Fuller Test:')
        dftest = adfuller(timeseries, autolag='AIC')
        dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
        for key,value in dftest[4].items():
            dfoutput['Critical Value (%s)'%key] = value
        if show_p_value:
            print(dfoutput)
            print('-------------------------')
        
        # Dickey Fuller test
        Dickey_Fuller_Hypothesis = {'Ho': 'Unit root is present in an autoregresstion model', 
                      'H1': 'Unit root is not present in an autoregresstion model'}
        
        # KPSS test
        KPSS_Hypothesis = {'Ho': 'Series is stationary', 
                      'H1': 'Series is not stationary'}
        if dfoutput['p-value'] <= 0.05:
            print('Results:')
            print(' By Dickey Fuller test:', 'Fail to reject Ho', "\n",
                  '=>', Dickey_Fuller_Hypothesis['H1'])
        if dfoutput['p-value'] <= 0.05:
            print('Results:')
            print(' By KPSS test:', 'Fail to reject Ho', "\n",
                  '=>', KPSS_Hypothesis['H1'])
        if dfoutput['p-value'] > 0.05:
            print('Results:')
            print(' By Dickey Fuller test:', 'Accept Ho', "\n",
                  '=>', Dickey_Fuller_Hypothesis['Ho'])
            
        if dfoutput['p-value'] > 0.05:
            print('Results:')
            print(' By KPSS test:', 'Accept Ho', "\n",
                  '=>', KPSS_Hypothesis['Ho'])
        # print('-------------------------')
            
    def make_ts_stationary(self, show_ma=False, show_na=False):
        # Smoothing by MA
        self.moving_avg = self.ts_log.rolling(window=self.window).mean()
        if show_ma: 
            plt.plot(self.ts_log)
            plt.plot(self.moving_avg, color='red')
            plt.title('Moving Average')
            
        ts_log_moving_avg_diff = self.ts_log - self.moving_avg
        
        
        if show_na: # shows the first window lenth of the data
            print(ts_log_moving_avg_diff.head(self.window))
        # Removing Null
        # self.ts_log_moving_avg_diff = null_removed
        return ts_log_moving_avg_diff.dropna()
    
    def exponentially_weighted_ma(self, show_plt=False):
        expwighted_avg = pd.DataFrame(self.ts_log).ewm(halflife=self.window).mean()
        
        if show_plt:
            plt.plot(self.ts_log)
            plt.plot(expwighted_avg, color='red')
        
        return (self.ts_log - expwighted_avg[self.target_values])
    
    def seasonality_along_with_trend(self, show_plt=False):
        ts_log_diff = self.ts_log - self.ts_log.shift()
        
        if show_plt:
            plt.plot(ts_log_diff)
            
        if (ts_log_diff.isnull().sum()) != 1:
            print('-------------------------')
            print('Sum not equal to 1')
            print('-------------------------')
        else:
            print('-------------------------')
            print('OK')
            print('-------------------------')
        return (ts_log_diff.copy().dropna())
    
    def seasonality_trend_residual_plot(self, show_plt=False):
        
        decomposition = seasonal_decompose(self.ts_log, period=self.window)
            
        trend = decomposition.trend
        seasonal = decomposition.seasonal
        residual = decomposition.resid
            
        if show_plt:
            plt.subplot(411)
            plt.plot(self.ts_log, label='Original')
            plt.legend(loc='best')
            plt.subplot(412)
            plt.plot(trend, label='Trend')
            plt.legend(loc='best')
            plt.subplot(413)
            plt.plot(seasonal,label='Seasonality')
            plt.legend(loc='best')
            plt.subplot(414)
            plt.plot(residual, label='Residuals')
            plt.legend(loc='best')
            plt.tight_layout()
        
        ts_log_decompose = self.ts_log - self.ts_log.shift()
        self.ts_log_diff = ts_log_decompose.dropna()
        return self.ts_log_diff
    
    def ACF_PACF(self, show_plt=False):
        if show_plt:
            plot_acf(self.ts_log_diff, lags =20)
            plot_pacf(self.ts_log_diff, lags =20)
            
            plt.show()
        else:
            pass
        
    def Auto_corr_and_Partial_Auto_corr(self, show_plt=False):
        if show_plt:
            #ACF and PACF plots:
            lag_acf = acf(self.ts_log_diff, nlags=self.window)
            lag_pacf = pacf(self.ts_log_diff, nlags=self.window, method='ols')
            
            #Plot ACF:    
            plt.subplot(121)    
            plt.plot(lag_acf)
            plt.axhline(y=0,linestyle='--',color='gray')
            plt.axhline(y=-1.96/np.sqrt(len(self.ts_log_diff)),linestyle='--',color='gray')
            plt.axhline(y=1.96/np.sqrt(len(self.ts_log_diff)),linestyle='--',color='gray')
            plt.title('Autocorrelation Function')
            
            #Plot PACF:
            plt.subplot(122)
            plt.plot(lag_pacf)
            plt.axhline(y=0,linestyle='--',color='gray')
            plt.axhline(y=-1.96/np.sqrt(len(self.ts_log_diff)),linestyle='--',color='gray')
            plt.axhline(y=1.96/np.sqrt(len(self.ts_log_diff)),linestyle='--',color='gray')
            plt.title('Partial Autocorrelation Function')
            plt.tight_layout()
        else:
            pass
            
    def split_data(self):
        X = self.ts
        self.train_size = int(len(X)*.80)
        self.test_size = int(len(X) - self.train_size)
        self.train_data, self.test_data = X[0:self.train_size], X[self.train_size:len(X)]
        
        self.history = [x for x in self.train_data]
        self.predictions = list()
        
        print('Data Splited:')
        print(' Train size:', self.train_size)
        print(' Test size:', self.test_size)
        print(' Total:', len(self.data))
        return self.train_size, self.test_size
            
    def fit(self, p,d,q,P,D,Q, evaluate=False):
        # walk-forward validation
        self.model = ARIMA(self.history, order=(p,d,q))
        self.model_fit = self.model.fit()
        
        if evaluate:
            
            predictions = list()
            history = self.history
            for t in range(len(self.test_data)):
                model = ARIMA(history, order=(p,d,q))
                model_fit = model.fit()
                output = model_fit.forecast()
                yhat = get_first_item(output)
                predictions.append(yhat)
                obs = self.test_data[t]
                history.append(obs)
            
            print('Model evaluation...')
            r2=r2_score(self.test_data, predictions)
            # MAPE
            mean_absolute_percentage_error = np.mean(np.abs(predictions - self.test_data)/np.abs(self.test_data))*100
            print('Forecast is off by {0}% and {1}% accurate'.format(
                np.round(mean_absolute_percentage_error,2), np.round(r2*100, 2)))
        
            print('--------Evaluation ended---------')
        
            
    def test_data_focast_evaluation(self):
        predictions = list()
        history = self.history
        for t in range(len(self.test_data)):
            model = ARIMA(history, order=(self.p, self.d, self.q))
            model_fit = model.fit()
            output = model_fit.forecast()
            yhat = get_first_item(output)
            predictions.append(yhat)
            obs = self.test_data[t]
            history.append(obs)
                
        plt.plot(self.test_data.index, predictions, label='Predicted')
        plt.plot(self.test_data.index, self.test_data, color='red', label='Observed')
        plt.legend()
    
    def model_evaluation(self):
        predictions = list()
        for t in range(len(self.test_data)):
            self.model = ARIMA(self.history, order=(self.p, self.d, self.q))
            self.model_fit = self.model.fit()
            output = self.model_fit.forecast()
            yhat = get_first_item(output)
            predictions.append(yhat)
            obs = self.test_data[t]
            self.history.append(obs)
            print('predicted=%f, Observed=%f' % (yhat, obs))
            
        # r2
        r2=r2_score(self.test_data, predictions)
        # MAPE
        mean_absolute_percentage_error = np.mean(np.abs(predictions - self.test_data)/np.abs(self.test_data))*100
        print('Forecast is off by {0}% and {1}% accurate'.format(
            np.round(mean_absolute_percentage_error,2), np.round(r2*100, 2)))
        
        model_test(self.test_data, predictions)
    
    
    def forecast(self, steps=1, show_plt=False):
        
        predictions = list()
        for t in range(self.test_size):
            if t>=1:
                self.model = ARIMA(self.history, order=(self.p, self.d, self.q))
                self.model_fit = self.model.fit()
                output = self.model_fit.forecast()
                yhat = get_first_item(output)
                predictions.append(yhat)
                obs = self.test_data[t]
                self.history.append(obs)

                
            else:
                output = self.model_fit.forecast()
                yhat = get_first_item(output)
                predictions.append(yhat)
                obs = self.test_data[t]
                self.history.append(obs)
                
        if show_plt:
            plt.plot(self.test_data.index, predictions, label='Forecast')
            plt.plot(self.test_data, color='red', label='Actual')
            plt.legend()
        return predictions
    
    def save(self, new_model_path):
        # saving the label_ids in to pickle
        file = new_model_path+'/'+'trainner.sav'
        
        if not os.path.exists(new_model_path):
            os.makedirs(new_model_path)
            
        with  open(file, 'wb') as f:
            pickle.dump(self.model_fit, f)
            
    def load_model(self, model_path, show_r2=False):
        # saving the label_ids in to pickle
        file = model_path+'/'+'trainner.sav'
        # with  open(file, 'rb') as f:
        loaded_model = pickle.load(open(file, 'rb'))
            
        if show_r2:
            predictions = loaded_model.get_forecast(steps=len(self.test_data))
            predictions = predictions.predicted_mean
            r2=r2_score(self.test_data, predictions)
            # MAPE
            mean_absolute_percentage_error = np.mean(np.abs(predictions - self.test_data)/np.abs(self.test_data))*100
            print('Forecast is off by {0}% and {1}% accurate'.format(
                np.round(mean_absolute_percentage_error,2), np.round(r2*100, 2)))
        return loaded_model
        
                
            
        
        
        

    
        
