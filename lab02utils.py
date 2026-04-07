import numpy as np
import glob
import matplotlib.pyplot as plt
import sklearn
import cvxopt
import math
import random
from copy import copy
cvxopt.solvers.options['show_progress'] = False

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
    
    return np.array([population[o] for o in order]), pareto_front

def crossover(p1, p2):
    alpha = 0.15
    c1 = alpha * p1 + (1-alpha) * p2
    c2 = (1-alpha) * p1 + alpha * p2
    return c1, c2

def mutate(solution):
    alpha = solution * 5
    alpha[alpha == 0] = 1e-3  # avoid zeros
    return np.random.dirichlet(alpha)

def selection(population, n_offspring):
    offspring = []
    for _ in range(n_offspring//2):
        p1, p2 = random.choices(population, k = 2)
        c1, c2 = crossover(p1, p2)
        offspring.append(mutate(c1))
        offspring.append(mutate(c2))
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