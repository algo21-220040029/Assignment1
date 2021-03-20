# Assignment1 - Stock price direction prediction by using SVM
Using the backtest framework created in last term and reproduce the codes of a essay.  
First introduce the struture of the backtest framework.  
backtest.py has three base class: Data, Strategy and Backtest   
Data is responsible for extracting data from local, and it only extract some necessay data in base class.  
Different inheritance classes can be created according to requirements to extract the data required by the strategy.  
Strategy is responsible to create factor matrix.  
StrategyFundamental.py and StrategyWQ.py has implemented special strategy class.  
It contains the functions these kind of strategy needed.  
And assignment1 is to creat a StrategySVM.py.  
Backtest.py is reponsible for backtesting which Considering the price limit 、the minimum number of shares to buy.  
And it can caculates different indicators of strategy.  
The reason for predicting the rise and fall of the market index is because many factors perform well when the market rises, but will be unsatisfactory when the market falls.   
So there is a way to judge the rise and fall of the market or even a bull market or a bear market.  
It is helpful for market timing or option hedging.   
The market data used in the original text is the South Korean market and the Hong Kong market, and I reproduced it on A shares.  
The data uses stocks in the entire market, plus the CSI All-Share Index, the S&P 500 Index, and the exchange rate of the U.S. dollar to the name currency.  
First of all, because of the co-moved effect in the stock market, PCA can be used to reduce the dimensionality of stock data to reduce the amount of training. From the results of PCA, it can be found that there is a stock that contributes 25% of the variance.  
The example of PCA:  
![image](https://user-images.githubusercontent.com/78793744/111875726-2d3e5b00-89d6-11eb-8e9e-a92e17a7c060.png)
Second step is to construct factors.  
The RDP values are determined based on three-day-lagged) values for the SP&500 and USDCNY, and one-day-lagged for the constituents of CSI.  
The reason of choosing RDP-3 values for the formers is that market index and exchange rate always have delayed-effects on the index values.  
Since the constituents serve as market comprising elements, the co-movements between the elements affect the market itself immediately. 
Therefore, a shorter lagged period is selected. The direction to forecast is the sign of one-day-ahead RDP, which is denoted as RDP+1.  
And the result is shown as follow:
![image](https://user-images.githubusercontent.com/78793744/111875626-cd47b480-89d5-11eb-826e-3898716ffe77.png)  
The accuracy is not very high, and in the essay:  
![image](https://user-images.githubusercontent.com/78793744/111875660-ef413700-89d5-11eb-9681-8dce6843ac15.png)
It can find the accuracy in essay is higher than the rusult based on A-shared market.  
So the effect of this method in mainland china is not so strong.  


