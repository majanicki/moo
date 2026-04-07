import numpy as np
import glob
import matplotlib.pyplot as plt
import sklearn
import cvxopt
import math
import random
cvxopt.solvers.options['show_progress'] = False
from lab02utils import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

stocks = {}
for stock_filename in glob.glob("bundle1/*.txt"):
    name, datapoints = load_stock_data(stock_filename)
    stocks[name] = datapoints
print(f"Loaded {len(stocks)} stocks")

stocks_predicted_ratios = {} # predictions for last day only
harmonic_order=35
period_len=201
stock_predicted_values = {}
for stock_name, stock_datapoints in stocks.items():
    predictions, err = harmonic_regression_prediction(stock_datapoints,harmonic_order, period_len)
    
    stocks_predicted_ratios[stock_name] = predictions[-1] / stock_datapoints[-1]
    x = np.arange(len(stock_datapoints))
    # plt.plot(x, stock_datapoints)
    px = np.arange(len(stock_datapoints), len(stock_datapoints) + 100)
    # plt.plot(px, predictions)
    stock_predicted_values[stock_name] = predictions
    # plt.axvline(len(stock_datapoints), linestyle="dotted")
    # plt.fill_between(px, predictions - err, predictions + err, alpha=0.3)
    # plt.title(stock_name)
    # plt.show()

m = []
stocks_expected_return = []
for stock in stocks:
    m.append(stocks[stock])
    stocks_expected_return.append(stocks_predicted_ratios[stock])
m = np.array(m)
stocks_covariance = np.cov(m)
stocks_expected_return = np.array(stocks_expected_return) - 1

n_samples = 100
wsm_x_returns = []
wsm_y_risks = []
wsm_solutions = []
for i in range(n_samples + 1):
    wcm_weight = 1 / n_samples * i
    w = wsm_solve(wcm_weight, stocks_expected_return, stocks_covariance)
    ereturn, risk = evaluate_solution(w, stocks_expected_return, stocks_covariance)
    wsm_x_returns.append(ereturn)
    wsm_y_risks.append(risk)
    wsm_solutions.append(w)
wsm_x_returns = np.array(wsm_x_returns)


threshold_step = (np.max(wsm_x_returns)-np.min(wsm_x_returns))/(n_samples-1)
ecm_x_returns = []
ecm_y_risks = []
ecm_solutions = []
for i in range(n_samples):
    t = i * threshold_step + np.min(wsm_x_returns)
    w = ecm_solve(t, stocks_expected_return, stocks_covariance)
    ereturn, risk = evaluate_solution(w, stocks_expected_return, stocks_covariance)
    ecm_x_returns.append(ereturn)
    ecm_y_risks.append(risk)
    ecm_solutions.append(w)


ideal_pareto_front = np.array(ecm_solutions)

population_history = evolve(200, 100, stocks_expected_return, stocks_covariance, False)
print(inverted_generational_distance(ideal_pareto_front, population_history[-1], stocks_expected_return, stocks_covariance))
print(hypervolume(population_history[-1], (0.1,1.5), stocks_expected_return, stocks_covariance))
