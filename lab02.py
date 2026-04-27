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
import seaborn as sns
import pandas as pd

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

island_data = []
standard_data = []
steady_data = []


for _ in range(10):
    a = evolve_dynamic(50, 150, stocks_expected_return, stocks_covariance, False)
    c = evolve(50, 150, stocks_expected_return, stocks_covariance, False)
    d = evolve_steady(50, 1500, stocks_expected_return, stocks_covariance, False)

    

    island_data.append(a)
    standard_data.append(c)
    steady_data.append(d)

debug2d_animated_3way([steady_data[-1], island_data[-1], steady_data[-1]],stocks_expected_return, stocks_covariance)
plt.figure(figsize=(10, 9))

methods = [("dynamic", island_data), ("standard", standard_data), ("steady", steady_data)]
for name, histories in methods:
    x = []
    y = []
    stds = []

    num_gens = len(histories[0])  # assume all runs same length
    for i in range(num_gens):
        x.append(histories[0][i][0])
        hvs = []
        for history in histories:
            hv = inverted_generational_distance(
                ecm_solutions,
                history[i][1],
                stocks_expected_return,
                stocks_covariance
            )
            hvs.append(hv)

        y.append(np.mean(hvs))
        stds.append(np.std(hvs))

    y = np.array(y)
    stds = np.array(stds)

    plt.plot(x, y, label=f"pop={name}")
    plt.fill_between(x, y - stds, y + stds, alpha=0.2)

plt.ylabel("Hypervolume")
plt.xlabel("Evaluation")
plt.title(f"IGD for NSGA-II (2D case)")
plt.legend()
plt.savefig("figs/new_methods.png")
plt.show()

# pop_sizes = [20, 30, 40, 50]
# gen_sizes = [100, 150, 200]
# n_repeats = 10
# populations_2d = []

# for pop_size in pop_sizes:
#     for gen_size in gen_sizes:
#         pops = []
#         for _ in range(n_repeats):
#             pops.append(evolve(pop_size, gen_size, stocks_expected_return, stocks_covariance, False))
#         populations_2d.append((pop_size, gen_size,pops))
# populations_3d = []

# ideal_pareto_front_2d = np.array(ecm_solutions)
# ideal_pareto_front_3d = []

# for pop_size in pop_sizes:
#     for gen_size in gen_sizes:
#         pops = []
#         for _ in range(n_repeats):
#             pops.append(evolve(pop_size, gen_size, stocks_expected_return, stocks_covariance, True))
#             ideal_pareto_front_3d = pops[-1][-1]
#         populations_3d.append((pop_size, gen_size,pops))


# def visualize_sensitivity(populations, pareto_front, name):
#     records = []

#     for pop_size, gen_size, pops in populations:
#         c = [
#             inverted_generational_distance(
#                 pareto_front,
#                 p[-1],
#                 stocks_expected_return,
#                 stocks_covariance
#             )
#             for p in pops
#         ]

#         records.append({
#             "Population": pop_size,
#             "Generations": gen_size,
#             "IGD": np.mean(c)
#         })

#     df = pd.DataFrame(records)

#     # Pivot into grid
#     pivot = df.pivot(index="Generations", columns="Population", values="IGD")

#     plt.figure(figsize=(10, 8))
#     sns.set_theme(style="whitegrid", font_scale=1.2)

#     ax = sns.heatmap(
#         pivot,
#         cmap="viridis",
#         annot=True,          # show values
#         fmt=".3f",
#         linewidths=0.5,
#         linecolor="gray",
#         cbar_kws={"label": "Sensitivity (IGD)"}
#     )

#     ax.set_title(f"Sensitivity Analysis: {name}", fontsize=16, pad=15)
#     ax.set_xlabel("Population Size")
#     ax.set_ylabel("Generation Size")

#     plt.tight_layout()
#     plt.savefig(f"figs/sensitivity_{name}.png", dpi=300)
#     plt.show()

# visualize_sensitivity(populations_3d, ideal_pareto_front_3d, "3D")
# visualize_sensitivity(populations_2d, ideal_pareto_front_2d, "2D")


# best_pops = []
# best_pop_size = 0
# best_gen_size = 0

# best_hv = 0
# for pop_size, gen_size, pops in populations_2d:
#     c = []
#     for p in pops:
#         hv = hypervolume(p[-1], (0.1,1.5), stocks_expected_return, stocks_covariance)
#         if hv > best_hv:
#             best_pops = pops 
#             best_hv = hv
#             best_pop_size = pop_size
#             best_gen_size = gen_size


# plt.figure(figsize=(10, 9))

# for pop_size, gen_size, pops in populations_2d:
#     if gen_size != gen_sizes[-1]:
#         continue
#     x = []
#     y = []
#     stds = []

#     num_gens = len(pops[0])  # assume all runs same length

#     for i in range(num_gens):
#         x.append(i)
#         hvs = []

#         for p in pops:
#             hv = hypervolume(
#                 p[i],
#                 (0.1, 1.5),
#                 stocks_expected_return,
#                 stocks_covariance
#             )
#             hvs.append(hv)

#         y.append(np.mean(hvs))
#         stds.append(np.std(hvs))

#     y = np.array(y)
#     stds = np.array(stds)

#     plt.plot(x, y, label=f"pop={pop_size}")
#     plt.fill_between(x, y - stds, y + stds, alpha=0.2)

# plt.ylabel("Hypervolume")
# plt.xlabel("Generation")
# plt.title(f"Change in hypervolume for NSGA-II (2D case, pop={best_pop_size})")
# plt.legend()
# plt.savefig("figs/hv_2d.png")
# plt.show()

# c = evaluate_population(best_pops[-1][-1], stocks_expected_return, stocks_covariance)
# nsga2_x_returns = -c[:,1]
# nsga2_y_risks = c[:,0]

# fig, ax = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(10,9))
# ax[0].scatter(wsm_x_returns, wsm_y_risks); ax[0].set_title("WSM")
# ax[1].scatter(ecm_x_returns, ecm_y_risks); ax[1].set_title("ECM"); ax[1].set_ylabel("Risk")
# ax[2].scatter(nsga2_x_returns, nsga2_y_risks); ax[2].set_title("NSGAII")
# plt.xlabel("Return")
# plt.savefig("figs/pareto_front_comp_2d.png")
# plt.show()


# debug2d(best_pops[-1], stocks_expected_return, stocks_covariance)
# debug3d(populations_3d[-1][-1][-1], stocks_expected_return, stocks_covariance)
# initial_solution = get_random_solution(3)

# x = []
# y = []
# z = []

# for i in range(30):
#     s = mutate(initial_solution)
#     x.append(s[0])
#     y.append(s[1])
#     z.append(s[2])

# fig = plt.figure(figsize=(10, 9))
# ax = fig.add_subplot(111, projection='3d')

# # Plot initial point
# ax.scatter(initial_solution[0], initial_solution[1], initial_solution[2],
#            label="initial", s=100)

# # Plot mutated points
# ax.scatter(x, y, z, label="Mutated")

# # Labels (optional but helpful)
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")
# ax.set_xlim(0, 1)
# ax.set_ylim(0, 1)
# ax.set_zlim(0, 1)
# plt.legend()
# plt.title("Mutation")
# plt.savefig("figs/mutation_proof.png")
# plt.show()

# p1 = get_random_solution(3)
# p2 = get_random_solution(3)


# x = []
# y = []
# z = []

# for i in range(30):
#     s1, s2 = crossover(p1, p2)
#     x.append(s1[0])
#     y.append(s1[1])
#     z.append(s1[2])
#     x.append(s2[0])
#     y.append(s2[1])
#     z.append(s2[2])

# fig = plt.figure(figsize=(10, 9))
# ax = fig.add_subplot(111, projection='3d')

# ax.scatter([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
#            label="initial", s=100)

# ax.scatter(x, y, z, label="Offspring")

# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")
# ax.set_xlim(0, 1)
# ax.set_ylim(0, 1)
# ax.set_zlim(0, 1)
# plt.legend()
# plt.title("Crossover operator")
# plt.savefig("figs/crossover_proof.png")
# plt.show()


# population_history = evolve(200, 100, stocks_expected_return, stocks_covariance, False)
# print(inverted_generational_distance(ideal_pareto_front, population_history[-1], stocks_expected_return, stocks_covariance))
# print(hypervolume(population_history[-1], (0.1,1.5), stocks_expected_return, stocks_covariance))
