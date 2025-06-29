import os
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import time

# --- Helper Functions ---

# Initialize a random population of routes
def initialize_population(N, num_cities):
    population = []
    for _ in range(N):
        route = list(range(num_cities))
        random.shuffle(route)
        population.append(route)
    return population

# Calculate distance matrix based on edge weight type and store it on distance array
def calculate_dist_matrix(coords, edge_weight_type):
    n = len(coords)
    dist = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            if edge_weight_type == "EUC_2D":
                d = calculate_euc2d(coords[i], coords[j])
            elif edge_weight_type == "ATT":
                d = calculate_att_distance(coords[i], coords[j])
            elif edge_weight_type == 'GEO':
                d = calculate_geo(coords[i], coords[j])
            else:
                d = calculate_euc2d(coords[i], coords[j])

            dist[i][j] = d
            dist[j][i] = d

    return dist

# Helper functions to calculate distance
def calculate_euc2d(p1, p2):
    dist = math.dist(p1, p2)
    return int(dist + 0.5)

def calculate_att_distance(p1, p2):
    xd = p1[0] - p2[0]
    yd = p1[1] - p2[1]
    r = math.sqrt((xd * xd + yd * yd) / 10.0)
    t = round(r)
    if t < r:
        return t + 1
    else:
        return t

def deg_min_to_rad(deg_min):
    deg = int(deg_min)
    min = deg_min - deg
    return math.pi * (deg + 5.0 * min / 3.0) / 180.0

def calculate_geo(p1, p2):
    R = 6378.388
    lat1 = deg_min_to_rad(p1[0])
    lon1 = deg_min_to_rad(p1[1])
    lat2 = deg_min_to_rad(p2[0])
    lon2 = deg_min_to_rad(p2[1])
    q1 = math.cos(lon1 - lon2)
    q2 = math.cos(lat1 - lat2)
    q3 = math.cos(lat1 + lat2)
    val = 0.5 * ((1 + q1) * q2 - (1 - q1) * q3)
    val = max(min(val, 1), -1)
    d = R * math.acos(val) + 1.0
    return int(d)

# Hamming distance between two routes
def hamming_distance(route1, route2):
    return sum(1 for i, j in zip(route1, route2) if i != j)

# Cost difference after 2-opt swap 
def delta_2opt_cost(route, i, k, dist):
    n = len(route)
    a, b = route[i - 1], route[i]
    c, d = route[k], route[(k + 1) % n]
    return dist[a][c] + dist[b][d] - dist[a][b] - dist[c][d]

# 2-opt swap function
def two_opt_swap(route, i, k):
    return route[:i] + route[i:k+1][::-1] + route[k+1:]

# Calculate cost of given route using distance matrix
def path_cost(route, dist_matrix):
    cost = 0.0
    n = len(route)
    for i in range(n):
        cost += dist_matrix[route[i]][route[(i+1) % n]]
    return cost

# Hybrid Simulated Annealing + 2-opt algorithm
def simulated_annealing_2opt(Ri, dist_matrix, D, T_init, cooling_rate, T_final):
    N = len(Ri)
    T = T_init
    current_route = Ri[:]
    current_cost = path_cost(current_route, dist_matrix)

    for _ in range(D):
        improved = False
        for i in range(1, N - 1):
            for j in range(i + 1, N):
                # delta_2opt_cost is used to check if the swap improves the cost
                delta_cost = delta_2opt_cost(current_route, i, j, dist_matrix)
                new_cost = current_cost + delta_cost
                if new_cost < current_cost:
                    current_route = two_opt_swap(current_route, i, j)
                    current_cost = new_cost
                    improved = True
                else:
                    if T > 0:
                        delta_val = current_cost - new_cost
                        if delta_val == 0:
                            continue
                        probability = math.exp((delta_val / T) * current_cost)  # Simulated Annealing acceptance probability
                        if random.random() < probability:
                            current_route = two_opt_swap(current_route, i, j)
                            current_cost = new_cost
                            improved = True
        if T > T_final:
            T *= cooling_rate
        # Improves execution time significantly 
        if not improved:
            break  

    return current_route

# Update route using SA-2-opt based on hamming distance from given leader
def update_route_equations(wolf, leader, dist,iter):
    # Calculate Hamming distances 
    hd_leader = hamming_distance(wolf, leader)
    max_hd = max(hd_leader, 1) 
    D = random.randint(1, max_hd)
    # Apply Simulated Annealing + 2-opt with initial temperature
    T_init = 100 * (0.95 ** iter)
    new_wolf = simulated_annealing_2opt(wolf, dist, D, T_init, cooling_rate=0.95, T_final=1)
    return new_wolf

# Genetic operations 
def genetic_operation(parent1, parent2):
    num_cities = len(parent1)

    # Ordered Crossover 
    r1, r2 = sorted(random.sample(range(num_cities), 2))
    offspring = [-1] * num_cities
    offspring[r1:r2] = parent1[r1:r2]
    fill_pos = r2
    for city in parent2:
        if city not in offspring:
            while offspring[fill_pos] != -1:
                fill_pos = (fill_pos + 1) % num_cities
            offspring[fill_pos] = city

    # Insertion Mutation
    a, b = sorted(random.sample(range(num_cities), 2))
    city = offspring.pop(b)
    offspring.insert(a, city)

    return offspring

# Process subpopulation with given leader
def process_population_group(subpopulation, leader, dist):
    N = len(subpopulation)

    # First update wolves using SA + 2-opt
    for i in range(N):
        subpopulation[i] = update_route_equations(subpopulation[i], leader, dist, iter=1)

    # Then apply genetic operators to generate offspring
    offspring = []
    random.shuffle(subpopulation)
    for i in range(0, N - 1, 2):
        parent1 = subpopulation[i]
        parent2 = subpopulation[i+1]
        child1 = genetic_operation(parent1, parent2)
        child2 = genetic_operation(parent2, parent1)
        offspring.extend([child1, child2])

    combined = subpopulation + offspring
    combined_sorted = sorted(combined, key=lambda x: path_cost(x, dist))

    # Lastly apply rank-based elimination: P_elim = rank / total
    survivors = []
    total = len(combined_sorted)
    for rank, individual in enumerate(combined_sorted):
        elimination_prob = rank / total
        if random.random() > elimination_prob:
            survivors.append(individual)
        if len(survivors) >= N:
            break

    while len(survivors) < N:
        survivors.append(random.choice(combined_sorted))

    return survivors[:N]

# --- Main GWO Algorithm ---

def i_gwo(coords, dist, N=50, M=20):
    num_cities = len(coords)
    population = initialize_population(N, num_cities)
    
    costs = [path_cost(w, dist) for w in population]
    
    # Initialize historical best alpha
    best_alpha = population[0]
    best_alpha_cost = costs[0]
    
    start_time = time.time()

    for t in range(M):
        # Calculate costs and sort population
        costs = [path_cost(w, dist) for w in population]
        sorted_idx = np.argsort(costs)
        population = [population[i] for i in sorted_idx]
        costs = [costs[i] for i in sorted_idx]

        alpha = population[0]
        alpha_cost = costs[0]

        # Update historical best alpha if improved
        if alpha_cost < best_alpha_cost:
            best_alpha = alpha
            best_alpha_cost = alpha_cost
        
        beta, delta = population[1], population[2]
        
        # Split population into three groups
        a = int(0.05 * N)
        b = int(0.40 * N)

        # Process population groups with different leader 
        population[:a] = process_population_group(population[:a], alpha, dist)
        population[a:b] = process_population_group(population[a:b], beta, dist)
        population[b:] = process_population_group(population[b:], delta, dist)

        # Generate new leaders using genetic operators
        new_alpha = genetic_operation(best_alpha, beta)
        new_beta = genetic_operation(best_alpha, delta)
        new_delta = genetic_operation(beta, delta)

        # Accept new leaders if they improve the cost
        new_alpha_cost = path_cost(new_alpha, dist)
        if new_alpha_cost < best_alpha_cost:
            best_alpha = new_alpha
            best_alpha_cost = new_alpha_cost
            population[0] = new_alpha

        new_beta_cost = path_cost(new_beta, dist)
        if new_beta_cost < path_cost(beta, dist):
            population[1] = new_beta
        
        new_delta_cost = path_cost(new_delta, dist)
        if new_delta_cost < path_cost(delta, dist):
            population[2] = new_delta

        # Print best cost  
        elapsed = time.time() - start_time
        if t + 1 in [1,2,10,20]:
            print(f"Iteration {t+1}: Best Cost = {best_alpha_cost:.2f}, Time elapsed = {elapsed:.2f} seconds")
        
    return best_alpha, best_alpha_cost, dist

# --- File I/O ---

def read_tsp(filename):
    coords = []
    edge_weight_type = "EUC_2D"  
    with open(filename, 'r') as f:
        start = False
        for line in f:
            if line.startswith("EDGE_WEIGHT_TYPE"):
                edge_weight_type = line.split(":")[1].strip()
            if "NODE_COORD_SECTION" in line:
                start = True
                continue
            if start:
                if line.strip() == "EOF":
                    break
                parts = line.strip().split()
                coords.append((float(parts[1]), float(parts[2])))
    return coords, edge_weight_type

def read_opt_route(filename):
    route = []
    with open(filename, 'r') as f:
        start = False
        for line in f:
            if "TOUR_SECTION" in line:
                start = True
                continue
            if start:
                if "-1" in line:
                    break
                parts = line.strip().split()
                for p in parts:
                    route.append(int(p) - 1)  
    return route

# --- Visualization ---

def plot_route(coords, route, title):
    x = [coords[i][0] for i in route + [route[0]]]
    y = [coords[i][1] for i in route + [route[0]]]
    plt.plot(x, y, marker='o')
    plt.title(title)
    plt.axis('equal')

# --- Main ---

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    files = [f for f in os.listdir(script_dir) if f.endswith(".tsp")]

    for tsp_file in files:
        name = tsp_file.replace(".tsp", "")
        print(f"\nSolving: {name}")
        coords, edge_weight_type = read_tsp(tsp_file)
        num_cities = len(coords)
        dist = calculate_dist_matrix(coords,edge_weight_type)

        # Set parameters based on number of cities
        runs = 1 if num_cities > 400 else 5
        M = 20 if num_cities > 400 else 20
        results = []

        # Run the GWO algorithm multiple times
        for run in range(runs):
            print(f"Run {run+1}/{runs}...")
            start_time = time.time()
            found_route, found_cost, _ = i_gwo(coords, dist, N=50, M=M)
            elapsed = time.time() - start_time
            results.append((found_cost, elapsed, found_route))

        costs, times, routes = zip(*results)
        avg_cost = np.mean(costs)
        avg_time = np.mean(times)
        best_idx = np.argmin(costs)
        best_cost = costs[best_idx]
        best_time = times[best_idx]
        best_route = routes[best_idx]

        print(f"\n{name} Summary:")
        print(f"Average Cost: {avg_cost:.2f}, Average Time: {avg_time:.2f} sec")
        print(f"Best Cost: {best_cost:.2f}, Time: {best_time:.2f} sec")

        # Plot best route and compare with optimal if available
        opt_path = name + ".opt.tour"
        if os.path.exists(opt_path):
            opt_route = read_opt_route(opt_path)
            opt_cost = path_cost(opt_route, dist)
            print(f"  Optimal Cost: {opt_cost:.2f}")

            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plot_route(coords, best_route, f"Best Found (Cost: {best_cost:.2f})")
            plt.subplot(1, 2, 2)
            plot_route(coords, opt_route, f"Optimal Route (Cost: {opt_cost:.2f})")
        else:
            print("No optimal route available.")
            plt.figure()
            plot_route(coords, best_route, f"Best Found (Cost: {best_cost:.2f})")

        plt.suptitle(f"TSP: {name}")
        plt.tight_layout()
        output_path = os.path.join(script_dir, f"{name}_comparison.png")
        plt.savefig(output_path)

if __name__ == "__main__":
    main()