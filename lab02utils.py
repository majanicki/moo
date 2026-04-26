import numpy as np
import glob
import matplotlib.pyplot as plt
import sklearn
import cvxopt
import math
import random
from copy import copy
cvxopt.solvers.options['show_progress'] = False
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import numpy as np
from scipy.spatial import cKDTree

def load_stock_data(filename):
    with open(filename, "r") as file:
        name : str
        n : int
        datapoints = []
        for i, line in enumerate(file):
            line = line.strip()
            if i == 0:
                name = line
            elif i == 1:
                n = int(line)
            else:
                _, point = line.split(" ")
                point = float(point)
                datapoints.append(point)
        assert(len(datapoints) == n)
        return name, np.array(datapoints)



def create_harmonic_elements(x, harmonic_order, period_len):
    intercept = np.ones_like(x)
    components = [intercept, x]
    for order in range(1, harmonic_order+1):
        zx = 2 * np.pi * order * x / period_len
        comp = np.sin(zx) + np.cos(zx)
        components.append(comp)
    components = np.column_stack(components)
    return components
    

def harmonic_regression_prediction(datapoints, harmonic_order, period_len):
    x = create_harmonic_elements(np.arange(len(datapoints)), harmonic_order, period_len)
    reg = sklearn.linear_model.BayesianRidge().fit(x, np.log(datapoints))
    err = np.std(datapoints - np.exp(reg.predict(x)))

    px = create_harmonic_elements(np.arange(len(datapoints), len(datapoints) + 100), harmonic_order, period_len) # go to next T
    return np.exp(reg.predict(px)), err


def wsm_solve(risk_weight, expected_return, covariance_matrix, normalize = True):
    n_factor_risk = 1.0
    n_factor_return = 1.0
    if normalize:
        wreturn = wsm_solve(0.0, expected_return, covariance_matrix, normalize=False)
        wrisk = wsm_solve(1.0, expected_return, covariance_matrix, normalize=False)
        max_return, max_risk = evaluate_solution(wreturn, expected_return, covariance_matrix)
        min_return, min_risk = evaluate_solution(wrisk, expected_return, covariance_matrix)
        n_factor_return = 1 / (max_return - min_return)
        n_factor_risk = 1 / (max_risk - min_risk)
    n = len(expected_return)
    Q = cvxopt.matrix(2 * risk_weight * n_factor_risk * covariance_matrix) # we have to flip for correct column layout I think
    c = cvxopt.matrix((1 - risk_weight) * expected_return * -1 * n_factor_return)
    b = cvxopt.matrix(1.0)
    A = cvxopt.matrix(np.ones(shape=(1,n)))

    G = cvxopt.matrix(-np.eye(n))
    h = cvxopt.matrix(np.zeros(n))

    sol = cvxopt.solvers.qp(Q, c, G, h, A, b)
    weights = np.array(sol['x']).flatten()
    return weights

def ecm_solve(epsilon, expected_return, covariance_matrix):
    n = len(expected_return)
    Q = cvxopt.matrix(2 * covariance_matrix) # we have to flip for correct column layout I think
    c = cvxopt.matrix(np.zeros(n))
    b = cvxopt.matrix(1.0)
    A = cvxopt.matrix(np.ones(shape=(1,n)))

    G = cvxopt.matrix(np.vstack((-expected_return,-np.eye(n))))
    h = np.zeros(n+1)
    h[0] = -epsilon
    h = cvxopt.matrix(h)

    sol = cvxopt.solvers.qp(Q, c, G, h, A, b)
    weights = np.array(sol['x']).flatten()
    return weights

def evaluate_solution(weights, expected_return, covariance_matrix):
    return np.sum(weights * (expected_return)), weights.T @ covariance_matrix @ weights


def evaluate_population(population, stocks_expected_return, stocks_covariance):
    risk = np.sum((population @ stocks_covariance) * population, axis=1)
    # risk = (risk - min_risk) / (max_risk - min_risk)
    ret = np.sum(population * stocks_expected_return, axis=1)
    # ret = (ret - min_return) / (max_return - min_return)
    return np.column_stack((risk, -ret))

def evaluate_population_3D(population, stocks_expected_return, stocks_covariance):
    risk = np.sum((population @ stocks_covariance) * population, axis=1)
    ret = np.sum(population * stocks_expected_return, axis=1)
    non_diverse = np.sum(population > 0.01, axis=1)
    return np.column_stack((risk, -ret, -non_diverse))

def get_crowding_distance(front, input_criteria, norm_min, norm_max):
    criteria = (input_criteria - norm_min)/(norm_max - norm_min)
    n = len(front)
    if n == 0:
        return []

    distances = [0.0] * n

    num_objectives = len(criteria[0])

    for m in range(num_objectives):
        sorted_idx = sorted(range(n), key=lambda i: criteria[front[i]][m])

        distances[sorted_idx[0]] = math.inf
        distances[sorted_idx[-1]] = math.inf

        for i in range(1, n - 1):
            prev_val = criteria[front[sorted_idx[i - 1]]][m]
            next_val = criteria[front[sorted_idx[i + 1]]][m]

            distances[sorted_idx[i]] += float((next_val - prev_val) / 2)
    return distances

            
def dominates(a, b):
    a = list(a)
    b = list(b)

    no_worse = all(x <= y for x, y in zip(a, b))
    strictly_better = any(x < y for x, y in zip(a, b))

    return no_worse and strictly_better

def ngsa2_sort(population, criteria):
    order = []
    pareto_front = []
    to_check = [i for i in range(len(population))]

    norm_min = 0
    norm_max = 0
    while len(to_check) > 0:
        non_dominated = []
        next_to_check = []
        for i in to_check:
            nd = True
            for j in to_check:
                if dominates(criteria[j], criteria[i]):
                    nd = False
                    break
            if nd:
                non_dominated.append(i)
            else:
                next_to_check.append(i)
        to_check = next_to_check
        if len(order) == 0:
            norm_min = criteria[non_dominated].min(axis=0)
            norm_max = criteria[non_dominated].max(axis=0)
        cd = get_crowding_distance(non_dominated, criteria, norm_min, norm_max)
        if len(order) == 0:
            pareto_front = non_dominated
        order += [x for _, x in sorted(zip(cd, non_dominated), key=lambda pair: pair[0], reverse=True)]
    
    return np.array([population[o] for o in order]), pareto_front, order

# def crossover(p1, p2):
#     alpha = random.gauss(0.0, 0.15)
#     c1 = alpha * p1 + (1-alpha) * p2
#     c2 = (1-alpha) * p1 + alpha * p2
#     if np.any(c1 < 0):
#         c1 = p1
#     if np.any(c2 < 0):
#         c2 = p2
#     return c1, c2

def crossover(p1, p2, index=15):
    u = random.random()

    if u < 0.5:
        beta = pow(2 * u, 1/(index + 1))
    else:
        beta = pow(1/(2 * (1 - u)), 1/(index + 1))

    c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
    c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)

    if np.any(c1 < 0):
        c1 = p1
    if np.any(c2 < 0):
        c2 = p2

    return c1, c2

# def crossover(p1, p2):
#     mask = np.random.binomial(1, 0.5, size=len(p1))

#     c1 = mask * p1 + (1 - mask) * p2
#     c2 = mask * p2 + (1 - mask) * p1
#     print(sum(c1), sum(c2))
#     return c1, c2

def mutate(solution, mstrength=50):
    alpha = solution * mstrength
    alpha[alpha == 0] = 1e-3  # avoid zeros
    return np.random.dirichlet(alpha)


def selection(population, n_offspring, index = 15, mstrength = 50):
    offspring = []
    for _ in range(n_offspring//2):
        p1, p2 = random.choices(population, k = 2)
        c1, c2 = crossover(p1, p2, index)
        offspring.append(mutate(c1, mstrength))
        offspring.append(mutate(c2, mstrength))
    return np.array(offspring)

def selection_close(population, n_offspring, index=15, mstrength=50):
    offspring = []

    dist_matrix = np.linalg.norm(population[:, None] - population[None, :], axis=2)

    np.fill_diagonal(dist_matrix, np.inf)

    for _ in range(n_offspring // 2):
        i = random.randrange(len(population))

        j = np.argmin(dist_matrix[i])

        p1, p2 = population[i], population[j]

        c1, c2 = crossover(p1, p2, index)
        offspring.append(mutate(c1, mstrength))
        offspring.append(mutate(c2, mstrength))

    return np.array(offspring)

def selection_elite(population, n_offspring, index=15, mstrength=50):
    offspring = []
    for _ in range(n_offspring//2):
        p1, p2 = random.choices(population[:int(len(population) * 0.1)], k = 2)
        c1, c2 = crossover(p1, p2, index)
        offspring.append(mutate(c1, mstrength))
        offspring.append(mutate(c2, mstrength))
    return np.array(offspring)

def get_random_solution(size):
    l = [0] + sorted(list(np.random.random(size - 1))) + [1]
    r = []
    for i in range(len(l) - 1):
        r.append(l[i+1] - l[i])
    return np.array(r)

def get_random_population(count, weight_len):
    r = []
    for _ in range(count):
        r.append(get_random_solution(weight_len))
    return np.array(r)





def evolve(pop_size, generations, stocks_expected_return, stocks_covariance, three_dimensional):

    evaluate_f = evaluate_population if not three_dimensional else evaluate_population_3D
    population = get_random_population(pop_size, len(stocks_expected_return))
    criteria = evaluate_f(population, stocks_expected_return, stocks_covariance)
    population, front, _ = ngsa2_sort(population, criteria)
    evals = pop_size
    population_history = [(evals, population)]
    for g in range(1, generations):

        offspring = selection(population, int(pop_size * 0.2))
        population = np.vstack((population, offspring))
        criteria = evaluate_f(population, stocks_expected_return, stocks_covariance)
        population, front, _ = ngsa2_sort(population, criteria)
        population = population[:pop_size]
        print(g, criteria[front[0]])
        evals += len(offspring)
        population_history.append((evals, population))

    return population_history

def evolve_island(pop_size, generations, stocks_expected_return, stocks_covariance, three_dimensional):

    evaluate_f = evaluate_population if not three_dimensional else evaluate_population_3D
    population_explore = get_random_population(pop_size, len(stocks_expected_return))
    population_exploit = get_random_population(pop_size, len(stocks_expected_return))

    criteria_explore = evaluate_f(population_explore, stocks_expected_return, stocks_covariance)
    population_explore, front, _ = ngsa2_sort(population_explore, criteria_explore)

    criteria_exploit = evaluate_f(population_exploit, stocks_expected_return, stocks_covariance)
    population_exploit, front, _ = ngsa2_sort(population_exploit, criteria_exploit)

    population_history_exploit = [population_exploit.copy()]
    population_history_explore = [population_explore.copy()]

    evals = pop_size * 2
    population_history = [(evals, np.vstack((population_explore, population_exploit))) ]

    n_migrants = 10
    
    for g in range(1, generations):

        offspring = selection(population_explore, int(pop_size * 0.2), index=5, mstrength=10)
        # population_explore = np.vstack((population_explore, offspring, population_history_exploit[-1][np.random.choice(pop_size, n_migrants)]))
        population_explore = np.vstack((population_explore, offspring, population_history_exploit[-1][:n_migrants]))
        criteria = evaluate_f(population_explore, stocks_expected_return, stocks_covariance)
        population_explore, front, _ = ngsa2_sort(population_explore, criteria)
        population_explore = population_explore[:pop_size]

        offspring = selection(population_exploit, int(pop_size * 0.2), index=30, mstrength=100)
        # population_exploit = np.vstack((population_exploit, offspring, population_history_explore[-1][np.random.choice(pop_size, n_migrants)]))
        population_exploit = np.vstack((population_exploit, offspring, population_history_explore[-1][:n_migrants]))
        criteria = evaluate_f(population_exploit, stocks_expected_return, stocks_covariance)
        population_exploit, front, _ = ngsa2_sort(population_exploit, criteria)
        population_exploit = population_exploit[:pop_size]

        print(g)
        population_history_explore.append(population_explore.copy())
        population_history_exploit.append(population_exploit.copy())
        evals += len(offspring) * 2
        population_history.append((evals, np.vstack((population_explore, population_exploit))) )

    return population_history

def evolve_steady(pop_size, generations, stocks_expected_return, stocks_covariance, three_dimensional):

    evaluate_f = evaluate_population if not three_dimensional else evaluate_population_3D
    population = get_random_population(pop_size, len(stocks_expected_return))
    criteria = evaluate_f(population, stocks_expected_return, stocks_covariance)
    population, front, order = ngsa2_sort(population, criteria)
    criteria = criteria[order]
    evals = pop_size
    population_history = [(evals, population)]
    for g in range(1, generations):

        offspring = selection_elite(population, 2)[1:]
        c_offspring = evaluate_f(offspring, stocks_expected_return, stocks_covariance)
        population = np.vstack((population, offspring))
        criteria = np.vstack((criteria, c_offspring))
        population, front, order = ngsa2_sort(population, criteria)
        criteria = criteria[order]
        population = population[:pop_size]
        criteria = criteria[:pop_size]
        print(g, criteria[front[0]])
        evals += len(offspring)
        population_history.append((evals, population))

    return population_history

def debug2d(population_history, stocks_expected_return, stocks_covariance):
    pareto_history_x = []
    pareto_history_y = []
    pareto_history_c = []
    for g, population in enumerate(population_history):
        _, population = population
        criteria = evaluate_population(population, stocks_expected_return, stocks_covariance)
        pareto_history_x += list(criteria[:, 1])
        pareto_history_y += list(criteria[:, 0])
        pareto_history_c += [g] * len(criteria[:, 0])
    plt.figure(figsize=(10, 9))
    plt.scatter(-np.array(pareto_history_x), pareto_history_y, c=pareto_history_c)
    plt.colorbar(label='Generation')
    plt.xlabel('Return')
    plt.ylabel('Risk')
    plt.savefig("figs/pop_history_2d.png")
    plt.show()

def debug3d(population_history, stocks_expected_return, stocks_covariance):
    pareto_history_x = []
    pareto_history_y = []
    pareto_history_z = []
    pareto_history_c = []
    for g, population in enumerate(population_history):
        criteria = evaluate_population_3D(population, stocks_expected_return, stocks_covariance)
        pareto_history_x += list(criteria[:, 1])
        pareto_history_y += list(criteria[:, 0])
        pareto_history_z += list(criteria[:, 2])
        pareto_history_c += [g] * len(criteria[:, 0])
    x = -np.array(pareto_history_x)
    y = np.array(pareto_history_y)
    z = -np.array(pareto_history_z)
    c = np.array(pareto_history_c)

    views = [
        (20, 45),
        (20, 135),
        (20, 225),
        (20, 315),
    ]

    for i, (elev, azim) in enumerate(views):
        fig = plt.figure(figsize=(10, 9))
        ax = fig.add_subplot(111, projection='3d')

        sc = ax.scatter([], [], [], s=20, alpha=0.7, c=[], cmap='viridis')

        ax.set_xlabel('Return')
        ax.set_ylabel('Risk')
        ax.set_zlabel('Number of significant weights')
        ax.set_xlim(np.min(x), np.max(x))
        ax.set_ylim(np.min(y), np.max(y))
        ax.set_zlim(np.min(z), np.max(z))

        ax.view_init(elev=elev, azim=azim)

        def update(frame):
            mask = (c == frame)
            sc._offsets3d = (x[mask], y[mask], z[mask])
            sc.set_array(c[mask])
            ax.set_title(f'Gen {frame} | elev={elev}, azim={azim}')
            return sc,

        anim = FuncAnimation(
            fig, update,
            frames=np.unique(c),
            interval=200,
            blit=False
        )

        anim.save(f"figs/pop_history_3d_view_{i}.gif", writer="pillow", fps=5)
        plt.close(fig)

def inverted_generational_distance(ideal_pareto_front, population,
                                   stocks_expected_return, stocks_covariance):

    criteria_population = evaluate_population(population,
                                              stocks_expected_return,
                                              stocks_covariance)
    criteria_pf = evaluate_population(ideal_pareto_front,
                                      stocks_expected_return,
                                      stocks_covariance)

    # normalization
    norm_min = criteria_pf.min(axis=0)
    norm_max = criteria_pf.max(axis=0)
    denom = np.where(norm_max - norm_min == 0, 1.0, norm_max - norm_min)

    criteria_population = (criteria_population - norm_min) / denom
    criteria_pf = (criteria_pf - norm_min) / denom

    # build KD-tree on population
    tree = cKDTree(criteria_population)

    # query nearest neighbor distances for all PF points at once
    distances, _ = tree.query(criteria_pf, k=1)

    return distances.mean()


def hypervolume(population, reference_point, stocks_expected_return, stocks_covariance):
    criteria_population = evaluate_population(population, stocks_expected_return, stocks_covariance)
    population, pareto_front, _ = ngsa2_sort(population, criteria_population)
    criteria_population = criteria_population[pareto_front]
    criteria_population = criteria_population[np.lexsort((criteria_population[:, 1], -criteria_population[:, 0]))]
    
    total_volume = 0
    for y, x in criteria_population:
        volume = (reference_point[0] - x) * (reference_point[1] - y)
        reference_point = (reference_point[0], y)
        total_volume += volume
    return total_volume

# Corssover - poprawić
# Dirichlet - udowodnić
# Lepsze ładniejsze confusion heatmap, 10 uruchomień
# Populacja mniejsza, żeby było widać różnicę


# Koewalujca:
# -  można mierzyć success rate, śledzić z której wyspy są rozwiązania
# - zrobić fajnie matematycznie
# - tyle samo ewluacji
# - specjalizacje powinny innaczej działać
# - dopieszczony ma być, dynamiczne population size i ma widać w wynikach, że jest lepiej
# - auc z convergence plot

# Fajny pomysł:
# - każdy population element mógłby trzymać drzewo genealogiczne
# - gradient na podstawie rodziców, lub archiwum

# Inny fajny pomysł:
# - steady state z NSGAII

# Inny fajny pomysł:
# - próba przewidywania zmiany z crowding distance

# Inny fajny pomysł:
# - dynamiczny population size

# POKAŻ WPROST, ŻE JEST IMPROVEMENT (tempo convergance)

# NSGAII:
# - restricated mating pool (krzyżujemy rozwiązania blisko siebie)