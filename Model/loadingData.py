import pandas_datareader as web

# loading data tUserWarningse
def loadData(startDate, endDate):
    df = web.DataReader('AAPL', data_source='yahoo', start=startDate, end=endDate)
    return df
# Show teh data
# print(df.shape)

## Visualize the closing price history
#plt.figure(figsize=(16,8))
#plt.title('Clise Price History')
#plt.plot(df['Close'])
#plt.xlabel('Date', fontsize=18)
#plt.ylabel('Close Price USD ($)', fontsize=18)
#plt.show()

# if __name__ = "__main__":
