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


pop_sizes = [100, 150, 200, 250]
gen_sizes = [50, 100, 150, 200]
n_repeats = 3
populations_2d = []

for pop_size in pop_sizes:
    for gen_size in gen_sizes:
        pops = []
        for _ in range(n_repeats):
            pops.append(evolve(pop_size, gen_size, stocks_expected_return, stocks_covariance, False))
        populations_2d.append((pop_size, gen_size,pops))
populations_3d = []

ideal_pareto_front_2d = np.array(ecm_solutions)
ideal_pareto_front_3d = []

for pop_size in pop_sizes:
    for gen_size in gen_sizes:
        pops = []
        for _ in range(n_repeats):
            pops.append(evolve(pop_size, gen_size, stocks_expected_return, stocks_covariance, True))
            ideal_pareto_front_3d = pops[-1][-1]
        populations_3d.append((pop_size, gen_size,pops))


def visualize_sensitivity(populations, pareto_front, name):
    values = []
    x_vals = []
    y_vals = []

    # compute sensitivity metric for each configuration
    for pop_size, gen_size, pops in populations:
        c = []
        for p in pops:
            c.append(
                inverted_generational_distance(
                    pareto_front,
                    p[-1],
                    stocks_expected_return,
                    stocks_covariance
                )
            )

        values.append(np.mean(c))
        x_vals.append(pop_size)
        y_vals.append(gen_size)

    x_unique = np.sort(np.unique(x_vals))
    y_unique = np.sort(np.unique(y_vals))

    x_index = {v: i for i, v in enumerate(x_unique)}
    y_index = {v: i for i, v in enumerate(y_unique)}

    grid = np.full((len(y_unique), len(x_unique)), np.nan)

    for xv, yv, val in zip(x_vals, y_vals, values):
        i = y_index[yv]
        j = x_index[xv]
        grid[i, j] = val

    plt.figure(figsize=(6, 5))
    plt.imshow(
        grid,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        extent=[
            x_unique.min(),
            x_unique.max(),
            y_unique.min(),
            y_unique.max()
        ],
    )

    plt.xticks(x_unique)
    plt.yticks(y_unique)
    plt.colorbar(label="Sensitivity (IGD)")
    plt.xlabel("Population size")
    plt.ylabel("Generation size")
    plt.title(f"Sensitivity {name}")
    plt.savefig(f"figs/sensitivity_{name}.png")
    plt.show()

visualize_sensitivity(populations_3d, ideal_pareto_front_3d, "3D")
visualize_sensitivity(populations_2d, ideal_pareto_front_2d, "2D")


best_pops = []
best_pop_size = 0
best_gen_size = 0

best_hv = 0
for pop_size, gen_size, pops in populations_2d:
    c = []
    for p in pops:
        hv = hypervolume(p[-1], (0.1,1.5), stocks_expected_return, stocks_covariance)
        if hv > best_hv:
            best_pops = pops 
            best_hv = hv
            best_pop_size = pop_size
            best_gen_size = gen_size


x = []
y = []
stds = []

for i in range(len(best_pops[0])):
    x.append(i)
    hvs = []
    for p in best_pops:
        hv = hypervolume(p[i], (0.1,1.5), stocks_expected_return, stocks_covariance)
        hvs.append(hv)
    y.append(np.mean(hvs))
    stds.append(np.std(hvs))
y = np.array(y)
stds = np.array(stds)
plt.plot(x, y)
plt.fill_between(x, y-stds, y+stds, color="lightblue", alpha = 0.3)
plt.ylabel("Hypervolume")
plt.xlabel("Generation")
plt.title(f"Change in hypervolume for NSGAII in 2D case (population size = {best_pop_size})")
plt.savefig(f"figs/hv_2d.png")
plt.show()

c = evaluate_population(best_pops[-1][-1], stocks_expected_return, stocks_covariance)
nsga2_x_returns = -c[:,1]
nsga2_y_risks = c[:,0]

fig, ax = plt.subplots(3, 1, sharex=True, sharey=True)
ax[0].scatter(wsm_x_returns, wsm_y_risks)
ax[1].scatter(ecm_x_returns, ecm_y_risks)
ax[2].scatter(nsga2_x_returns, nsga2_y_risks)
plt.savefig("figs/pareto_front_comp_2d.png")
plt.show()
debug2d(best_pops[-1], stocks_expected_return, stocks_covariance)
debug3d(populations_3d[-1][-1][-1], stocks_expected_return, stocks_covariance)

# population_history = evolve(200, 100, stocks_expected_return, stocks_covariance, False)
# print(inverted_generational_distance(ideal_pareto_front, population_history[-1], stocks_expected_return, stocks_covariance))
# print(hypervolume(population_history[-1], (0.1,1.5), stocks_expected_return, stocks_covariance))
