import numpy as np
import glob
import matplotlib.pyplot as plt
import sklearn
import cvxopt
import math
import random
cvxopt.solvers.options['show_progress'] = False
from lab02utils import *

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


# eigvals = np.linalg.eigvalsh(stocks_covariance)
# min_risk = eigvals.min()
# max_risk = eigvals.max()
# max_return = np.max(stocks_expected_return)
# min_return = np.min(stocks_expected_return)

# print(min_return, max_return, min_risk, max_risk)

pop_size = 250
generations = 100

population = get_random_population(pop_size, len(stocks))
criteria = evaluate_population(population, stocks_expected_return, stocks_covariance)
population, front = ngsa2_sort(population, criteria)

population, front = ngsa2_sort(population, criteria)
pareto_history_x = list(criteria[front, 1])
pareto_history_y = list(criteria[front, 0])
pareto_history_z = [0] * len(criteria[front, 0])
final_nsga_risks = []
final_nsga_returns = []


for g in range(1, generations):

    offspring = selection(population, int(pop_size * 0.2))
    population = np.vstack((population, offspring))
    criteria = evaluate_population(population, stocks_expected_return, stocks_covariance)

    population, front = ngsa2_sort(population, criteria)

    population = population[:pop_size]
    final_nsga_risks = list(criteria[front, 0])
    final_nsga_returns = list(criteria[front, 1])
    pareto_history_x += list(criteria[front, 1])
    pareto_history_y += list(criteria[front, 0])
    pareto_history_z += [g] * len(criteria[front, 0])    
    print(g, criteria[front[0]])

plt.scatter(-np.array(pareto_history_x), pareto_history_y, c=pareto_history_z)
plt.colorbar(label='Z value')
plt.xlabel('Return')
plt.ylabel('Risk')
plt.show()
final_nsga_returns = -np.array(final_nsga_returns)

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


threshold_step = (np.max(wsm_x_returns)-np.min(wsm_x_returns))/n_samples
ecm_x_returns = []
ecm_y_risks = []
ecm_solutions = []
for i in range(n_samples + 1):
    t = i * threshold_step + np.min(wsm_x_returns)
    w = ecm_solve(t, stocks_expected_return, stocks_covariance)
    ereturn, risk = evaluate_solution(w, stocks_expected_return, stocks_covariance)
    ecm_x_returns.append(ereturn)
    ecm_y_risks.append(risk)
    ecm_solutions.append(w)
plt.scatter(final_nsga_returns, final_nsga_risks, label="NSGAII", marker=".")
plt.scatter(ecm_x_returns, ecm_y_risks, label="ECM", marker=".", alpha=0.5)
plt.scatter(wsm_x_returns, wsm_y_risks, label="WSM", marker=".", alpha=0.5)
plt.legend()
plt.show()
