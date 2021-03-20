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
