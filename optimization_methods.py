import math, itertools, time
import pandas as pd
import matplotlib.pyplot as plt

locations = [('Depot - Galle General Hospital', 0.0, 0.0), ('Hikkaduwa Pharmacy', 16, 18), ('Ambalangoda Medical Center', 28, 30), ('Elpitiya Clinic', 24, 10), ('Baddegama Pharmacy', 12, 8), ('Akuressa Rural Hospital', -8, 18), ('Imaduwa Medical Point', 4, -10), ('Koggala Pharmacy', 10, -18), ('Weligama Health Store', 22, -24), ('Deniyaya Clinic', -18, 12), ('Yakkalamulla Pharmacy', -6, -6), ('Udugama Rural Dispensary', -20, 2), ('Karapitiya Sub Depot', 2, 4)]
coords = [(x, y) for _, x, y in locations]

# Euclidean distance matrix
D = [[0 if i == j else round(math.hypot(coords[i][0]-coords[j][0], coords[i][1]-coords[j][1]), 2)
      for j in range(len(coords))] for i in range(len(coords))]


def held_karp(dist_matrix):
    n = len(dist_matrix)
    C = {}
    parent = {}
    for k in range(1, n):
        mask = 1 << (k - 1)
        C[(mask, k)] = dist_matrix[0][k]
        parent[(mask, k)] = 0

    for subset_size in range(2, n):
        for subset in itertools.combinations(range(1, n), subset_size):
            bits = 0
            for b in subset:
                bits |= 1 << (b - 1)
            for k in subset:
                prev = bits & ~(1 << (k - 1))
                best = float('inf')
                best_prev = None
                for m in subset:
                    if m == k:
                        continue
                    val = C[(prev, m)] + dist_matrix[m][k]
                    if val < best:
                        best = val
                        best_prev = m
                C[(bits, k)] = best
                parent[(bits, k)] = best_prev

    bits = (1 << (n - 1)) - 1
    best = float('inf')
    last = None
    for k in range(1, n):
        val = C[(bits, k)] + dist_matrix[k][0]
        if val < best:
            best = val
            last = k

    seq = []
    subset = bits
    k = last
    while subset:
        seq.append(k)
        pk = parent[(subset, k)]
        subset &= ~(1 << (k - 1))
        k = pk
        if k == 0:
            break

    route = [0] + list(reversed(seq)) + [0]
    return round(best, 2), route


def route_cost(route, dist_matrix):
    return round(sum(dist_matrix[route[i]][route[i+1]] for i in range(len(route)-1)), 2)


def nearest_neighbor(dist_matrix):
    unvisited = set(range(1, len(dist_matrix)))
    route = [0]
    current = 0
    while unvisited:
        nxt = min(unvisited, key=lambda j: dist_matrix[current][j])
        route.append(nxt)
        unvisited.remove(nxt)
        current = nxt
    route.append(0)
    return route


def two_opt(route, dist_matrix):
    best = route[:]
    improved = True
    while improved:
        improved = False
        best_cost = route_cost(best, dist_matrix)
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best) - 1):
                if j - i == 1:
                    continue
                candidate = best[:i] + best[i:j][::-1] + best[j:]
                cand_cost = route_cost(candidate, dist_matrix)
                if cand_cost < best_cost:
                    best, best_cost = candidate, cand_cost
                    improved = True
    return best


if __name__ == '__main__':
    t0 = time.perf_counter()
    exact_cost, exact_route = held_karp(D)
    exact_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    nn_route = nearest_neighbor(D)
    heuristic_route = two_opt(nn_route, D)
    heuristic_time = time.perf_counter() - t0
    heuristic_cost = route_cost(heuristic_route, D)

    print('Exact cost:', exact_cost)
    print('Exact route:', exact_route)
    print('Exact runtime:', round(exact_time, 6), 's')
    print('Heuristic cost:', heuristic_cost)
    print('Heuristic route:', heuristic_route)
    print('Heuristic runtime:', round(heuristic_time, 6), 's')
