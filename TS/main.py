#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from t_series import TimeSeries_analysis
import pandas as pd

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6

# Load data
LOADED_DATA = pd.read_csv('reg_fin_df.csv')
# convet data to ts data
TIME_SERIES = TimeSeries_analysis(data=LOADED_DATA, index_tearget='Month', target_values='OdoReading', window=12)
    
class ForcastModel:
    def __init__(self, window, Vehicle_registration):
        global TIME_SERIES
        print('--------- Forcating class initiation ----------')
        # plot data
        TIME_SERIES.data_plt(show_ts=True)
        # pdq estimation
        p,d,q, P,D,Q = TIME_SERIES.params_estimation()
        self.p, self.d, self.q = p, d, q
        self.P , self.D, self.Q = P, D, Q
        self.window = window
        
        self.new_model_path = "trained_models/" + Vehicle_registration
        self.Vehicle_registration = Vehicle_registration
        
        TIME_SERIES.seasonality_trend_residual_plot(show_plt=False)
        TIME_SERIES.ACF_PACF(show_plt=True)
        
        
    def stationarity_test(self):
        # stationarity
        TIME_SERIES.stationarity(show_ts=False)
        print('Stationarity test...')
        TIME_SERIES.stationarity_testing(show_p_value=True, show_rolmean_rolstd=False)
        # make_ts_stationary
        self.ts_log_moving_avg_diff = TIME_SERIES.make_ts_stationary(show_ma=False, show_na=False)
        # Test stationarity
        print('Stationarity test...')
        TIME_SERIES.stationarity_testing(show_p_value=True, show_rolmean_rolstd=True, data=self.ts_log_moving_avg_diff)
        
        print('-------------------------')
        self.ts_log_ewma_diff = TIME_SERIES.exponentially_weighted_ma(show_plt=False)
        
        if self.ts_log_ewma_diff is not None:
            print('Stationarity test...')
            TIME_SERIES.stationarity_testing(show_p_value=True, show_rolmean_rolstd=True, data=self.ts_log_ewma_diff)
        else:
            raise Exception('ts_log_ewma_diff empty')
            
            
    def split_data(self):
        _, __ = TIME_SERIES.split_data()
        
    def fit(self):
        TIME_SERIES.fit(self.p, self.d, self.q, self.P , self.D, self.Q, evaluate=True)
        
    def forecast(self):
        return TIME_SERIES.forecast(steps=self.window, show_plt=True)
        
        
    # -------------------------------------------unsed
    def model_evaluation(self):
        # test data v.s Predicted
        TIME_SERIES.test_data_focast_evaluation()
        # mode r2
        TIME_SERIES.model_evaluation()
        
    def save(self):
        TIME_SERIES.save(new_model_path=self.new_model_path)
        
    def load_model(self):
         return TIME_SERIES.load_model(self.new_model_path, show_r2=False)

# def ARIMA_focast(model):
#     model
#     self.model_fit = self.model.fit()
#     output = self.model_fit.forecast()
#     yhat = output[0][0]
#     predictions.append(yhat)
#     obs = self.test_data[t]
#     self.history.append(obs)
    
    
    
    # return predictions
def main(registration, period=12):
    # Loading model class
    mode_class = ForcastModel(window=period, Vehicle_registration=registration)
    print('-------------Splitting data---------------')
    # model split data
    mode_class.split_data()
    print('-------------Model fit---------------')
    # fit model
    mode_class.fit()
    
    #model forecast
    predicted = mode_class.forecast()
    predicted = [int(x) for x in predicted]
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print(predicted)
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@----')
    
    # save model
    mode_class.save()
    
    # Load model
    model = mode_class.load_model()
    
    # ARIMA_focast(model)
    
    # forecast_object = model.get_forecast(steps=period)
    # mean = forecast_object.predicted_mean
    # mean = [int(x) for x in mean]
    # print('//////////////////')
    # print(mean)
    # print('------------------')
    

       
    # # print('-------------------------')
    # ts_log_diff = TIME_SERIES.seasonality_along_with_trend(show_plt=False)
    # print('ts_log_diff....', ts_log_diff)
    # if ts_log_diff is not None:
    #     print('Stationarity test...')
    #     TIME_SERIES.stationarity_testing(show_p_value=True, show_rolmean_rolstd=True, data=ts_log_diff)
    # else:
    #     raise Exception('ts_log_diff empty')
        
    # Trend, seasonality and residual plot
    # ts_log_decompose = TIME_SERIES.seasonality_trend_residual_plot(show_plt=False)
        
    # if ts_log_diff is not None:
    #     print('Stationarity test...')
    #     TIME_SERIES.stationarity_testing(show_p_value=True, show_rolmean_rolstd=True, data=ts_log_decompose)
    # else:
    #     raise Exception('ts_log_decompose empty')
    # print('-------------------------')
    
    # # ACF: Checks for correlation with the previous observations
    # # PACF: Checks for correlation with the residuals
    
    # print('-------------------------info_')
    # p,d,q, P,D,Q = TIME_SERIES.params_estimation()
    # print('-------------------------***')

    # #Auto regresion plots
    # TIME_SERIES.Auto_corr_and_Partial_Auto_corr(show_plt=True)
    # ARIMA model fit
    # time_series.ARIMA_parms(data=ts_log_decompose, p=p,d=d,q=q)
    # time_series.ARIMA_fit(show_predictions=True, p=p,d=d,q=q)
    
    # time_series.ARIMA_predict(data=ts_log_decompose[])
    
    
    # time_series.forecast(steps=1, show_plt=True)
    
    
if __name__ == "__main__":
    while True:
        try:
            reg = str(input("Enter vehicle registration:\n", )).upper()
        except Exception as e:
            raise e
        break
    
    while True:
        try:
            window = int(input("Enter period(in months):\n", ))
            break
        except Exception as e:
            if str(e)[:38] == 'invalid literal for int() with base 10':
                print('period must be a number!!!')
                pass
            else:
                raise e
        
    main(registration=reg, period=window)

