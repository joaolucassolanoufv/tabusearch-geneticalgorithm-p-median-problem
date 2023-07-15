import numpy as np
import random
import time
from collections import deque

def read_ORLibrary_input_file(file_path):
    try:
        with open(file_path, "r") as f:
            first_line = f.readline().strip().split(' ')
            num_vertices = int(first_line[0])
            p = int(first_line[2])

            cost_matrix = np.matrix(np.ones((num_vertices, num_vertices)) * np.inf)

            for line in f:
                line = line.strip().split(' ')
                cost_matrix[int(line[0])-1, int(line[1])-1] = int(line[2])
                cost_matrix[int(line[1])-1, int(line[0])-1] = int(line[2])

            for i in range(0, num_vertices):
                cost_matrix[i, i] = 0

            for k in range(0, num_vertices):
                for i in range(0, num_vertices):
                    for j in range(0, num_vertices):
                        if cost_matrix[i, j] > cost_matrix[i, k] + cost_matrix[k, j]:
                            cost_matrix[i, j] = cost_matrix[i, k] + cost_matrix[k, j]

            return num_vertices, p, cost_matrix

    except IOError:
        return None

def calculate_solution_cost(num_vertices, p, cost_matrix, medians):
    total_cost = 0
    for i in range(num_vertices):
        min_distance = np.inf
        for j in medians:
            min_distance = min(min_distance, cost_matrix[i, j])
        total_cost += min_distance

    return total_cost

def generate_greedy_solution(num_vertices, p, cost_matrix):
    medians = []
    remaining_vertices = set(range(num_vertices))

    for _ in range(p):
        min_total_cost = np.inf
        best_median = None

        for vertex in remaining_vertices:
            medians.append(vertex)
            total_cost = calculate_solution_cost(num_vertices, p, cost_matrix, medians)

            if total_cost < min_total_cost:
                min_total_cost = total_cost
                best_median = vertex

            medians.pop()

        medians.append(best_median)
        remaining_vertices.remove(best_median)

    return medians

def generate_random_solution(num_vertices, p):
    solution = []
    for _ in range(p):
        gene = random.randint(0, num_vertices - 1)
        solution.append(gene)
    return solution

def generate_neighbor_solutions(num_vertices, solution, k = 5, r = 1):
    neighbor = solution.copy()
    p = len(set(neighbor))
    neighbors = []

    for i in range(p):
        remaining_vertices = set(range(num_vertices)) - set(neighbor)
        for _ in range(k):
            for _ in range(r):
                chosen_vertice = random.choice(list(remaining_vertices))

                neighbor[i] = chosen_vertice

                neighbors.append(neighbor)

                remaining_vertices.remove(chosen_vertice)
            neighbor = solution.copy()


    return neighbors

def local_search(num_vertices, p, cost_matrix, solution, opt_cost):
    best_solution = solution.copy()
    best_cost = calculate_solution_cost(num_vertices, p, cost_matrix, solution)

    improvement = True
    while improvement:
        improvement = False
        neighbors = generate_neighbor_solutions(num_vertices, solution)

        for neighbor in neighbors:
            neighbor_cost = calculate_solution_cost(num_vertices, p, cost_matrix, neighbor)                
            if neighbor_cost < best_cost:
                best_solution = neighbor
                best_cost = neighbor_cost
                if best_cost == opt_cost:
                    return best_solution, best_cost
                improvement = True
                break

    return best_solution, best_cost

def tabu_search(num_vertices, p, cost_matrix, solution, tabu_list_size, max_iterations, opt_cost):
    best_solution = solution.copy()
    best_cost = calculate_solution_cost(num_vertices, p, cost_matrix, solution)
    tabu_list = deque(maxlen=tabu_list_size)

    current_iteration = 0
    start_time = time.time()
    while current_iteration < max_iterations and best_cost > opt_cost and (time.time() - start_time) <= 900:
        neighbors = generate_neighbor_solutions(num_vertices, solution)

        candidate_solutions = []
        for neighbor in neighbors:
            if neighbor not in tabu_list:
                neighbor_cost = calculate_solution_cost(num_vertices, p, cost_matrix, neighbor)
                candidate_solutions.append((neighbor, neighbor_cost))

        if len(candidate_solutions) == 0:
            break

        candidate_solutions.sort(key=lambda x: x[1])
        next_solution, next_cost = candidate_solutions[0]

        if next_cost < best_cost:
            best_solution = next_solution
            best_cost = next_cost
        current_iteration += 1

        solution = next_solution
        tabu_list.append(next_solution)

    return local_search(num_vertices, p, cost_matrix, best_solution, opt_cost)

def initialize_population(num_vertices, p, population_size):
    population = []
    for _ in range(population_size):
        individual = generate_random_solution(num_vertices, p)
        population.append(individual)
    return population

def crossover(parent1, parent2, crossover_prob = 0.85):
    child1 = parent1.copy()
    child2 = parent2.copy()

    if random.random() < crossover_prob:
        parent1_genes = set(parent1)

        for i, gene in enumerate(parent2):
            if gene not in parent1_genes:
                child1[i] = gene
                child2[i] = parent1[i]

    return child1, child2

def genetic_algorithm(num_vertices, p, cost_matrix, population_size, max_generations, opt_cost, inicial_solution, random_population = False):
    population = []

    if random_population:
        population = initialize_population(num_vertices, p, population_size)
    else:
        neighbor_solutions = generate_neighbor_solutions(
            num_vertices, 
            inicial_solution, 
            int((population_size)/(3*p))
        )
        for solution in neighbor_solutions:
            population.append(solution)

        for _ in range(int(population_size - len(population))):
            population.append(generate_random_solution(num_vertices, p))
    
    random.shuffle(population)
    
    best_solution = inicial_solution
    best_cost = calculate_solution_cost(num_vertices, p, cost_matrix, inicial_solution)

    num_parents = int(population_size / 2)
    if num_parents%2 == 1:
        num_parents += 1

    start_time = time.time()
    for _ in range(max_generations):
        if best_cost == opt_cost:
            break

        if (time.time() - start_time) > 900:
            break

        offspring = []
        fitness_values = []

        for solution in population:
            

            cost = calculate_solution_cost(num_vertices, p, cost_matrix, solution)
            fitness = 1 / (cost + 1e-6)  # Maior fitness para soluções com menor custo
            fitness_values.append(fitness)

            if cost < best_cost:
                best_solution = solution.copy()
                best_cost = cost

        parents = random.choices(population, weights=fitness_values, k=num_parents)

        for i in range(0, num_parents, 2):
            parent1 = parents[i]
            parent2 = parents[i + 1]

            child1, child2 = crossover(parent1, parent2)

            mutation_prob = 0.05
            if random.random() < mutation_prob:
                gene_idx = random.randint(0, p - 1)
                new_gene = random.randint(0, num_vertices - 1)
                while new_gene in child1:
                    new_gene = random.randint(0, num_vertices - 1)
                child1[gene_idx] = new_gene

            if random.random() < mutation_prob:
                gene_idx = random.randint(0, p - 1)
                new_gene = random.randint(0, num_vertices - 1)
                while new_gene in child2:
                    new_gene = random.randint(0, num_vertices - 1)
                child2[gene_idx] = new_gene

            offspring.extend([child1, child2])

        population.extend(offspring)
        population = sorted(population, key=lambda x: calculate_solution_cost(num_vertices, p, cost_matrix, x))
        population = population[:population_size]

    return local_search(num_vertices, p, cost_matrix, best_solution, opt_cost)

def calculate_optimal_value(cost, opt_cost):
    return round(100 * (1 - (cost - opt_cost) / opt_cost), 2)

uncap_cost_solution_opt = {
    1: 5819,
    2: 4093,
    3: 4250,
    4: 3034,
    5: 1355,
    6: 7824,
    7: 5631,
    8: 4445,
    9: 2734,
    10: 1255,
    11: 7696,
    12: 6634,
    13: 4374,
    14: 2968,
    15: 1729,
    16: 8162,
    17: 6999,
    18: 4809,
    19: 2845,
    20: 1789,
    21: 9138,
    22: 8579,
    23: 4619,
    24: 2961,
    25: 1828,
    26: 9917,
    27: 8307,
    28: 4498,
    29: 3033,
    30: 1989,
    31: 10086,
    32: 9297,
    33: 4700,
    34: 3013,
    35: 10400,
    36: 9934,
    37: 5057,
    38: 11060,
    39: 9423,
    40: 5128
}

uncap_cost_solution_greedy = {}
uncap_cost_solution_local_search = {}
uncap_cost_solution_tabu_search = {}
uncap_cost_solution_genetic_algorithm = {}
uncap_cost_solution_ant_colony = {}

uncap_solution_solution_greedy = {}
uncap_solution_solution_local_search = {}
uncap_solution_solution_tabu_search = {}
uncap_solution_solution_genetic_algorithm = {}
uncap_solution_solution_ant_colony = {}

uncap_otimal_solution_greedy= {}
uncap_otimal_solution_local_search = {}
uncap_otimal_solution_tabu_search = {}
uncap_otimal_solution_genetic_algorithm = {}
uncap_otimal_solution_ant_colony = {}

for i in range(20, 26):
    num_vertices, p, cost_matrix = read_ORLibrary_input_file('OR-Library/pmed' + str(i) + '.txt')

    print("Opt Cost pmed" + str(i) + ": ", uncap_cost_solution_opt[i])
    print("\n")

    start_time = time.time()
    greedy_solution = generate_greedy_solution(num_vertices, p, cost_matrix)
    end_time = time.time()
    greedy_cost = calculate_solution_cost(num_vertices, p, cost_matrix, greedy_solution)
    uncap_otimal_solution_greedy[i] = calculate_optimal_value(greedy_cost, uncap_cost_solution_opt[i])

    print("Greedy Cost pmed" + str(i) + ": ", greedy_cost)
    print("Optimal Value: ", uncap_otimal_solution_greedy[i], "%")
    print("Time taken for Greedy Algorithm: ", round(end_time - start_time, 2), "seconds")
    print("\n")

    start_time = time.time()
    local_search_solution, local_search_cost = local_search(num_vertices, p, cost_matrix, greedy_solution, uncap_cost_solution_opt[i])
    end_time = time.time()
    uncap_otimal_solution_local_search[i] = calculate_optimal_value(local_search_cost, uncap_cost_solution_opt[i])

    print("Local Search cost pmed" + str(i) + ": ", local_search_cost)
    print("Optimal Value: ", uncap_otimal_solution_local_search[i], "%")
    print("Time taken for Local Search: ", round(end_time - start_time, 2), "seconds")
    print("\n")

    for tabu_list_size in [20, 30, 40]:
        print("TABU LIST SIZE: " + str(tabu_list_size))
        start_time = time.time()
        tabu_solution, tabu_cost = tabu_search(num_vertices, p, cost_matrix, local_search_solution, tabu_list_size=tabu_list_size, max_iterations=100, opt_cost=uncap_cost_solution_opt[i])
        end_time = time.time()
        uncap_otimal_solution_tabu_search[i] = calculate_optimal_value(tabu_cost, uncap_cost_solution_opt[i])

        print("Tabu Search cost pmed" + str(i) + ": ", tabu_cost)
        print("Optimal Value: ", uncap_otimal_solution_tabu_search[i], "%")
        print("Time taken for Tabu Search: ", round(end_time - start_time, 2), "seconds")
        print("\n")

        start_time = time.time()
        genetic_solution, genetic_cost = genetic_algorithm(num_vertices, p, cost_matrix, population_size=200, max_generations=200, opt_cost=uncap_cost_solution_opt[i], inicial_solution=tabu_solution)
        end_time = time.time()
        uncap_otimal_solution_genetic_algorithm[i] = calculate_optimal_value(genetic_cost, uncap_cost_solution_opt[i])

        print("Genetic Algorithm (TABU) cost pmed" + str(i) + ": ", genetic_cost)
        print("Optimal Value: ", uncap_otimal_solution_genetic_algorithm[i], "%")
        print("Time taken for Genetic Algorithm (TABU): ", round(end_time - start_time, 2), "seconds")
        print("\n")

    uncap_cost_solution_greedy[i] = greedy_cost
    uncap_cost_solution_local_search[i] = local_search_cost
    uncap_cost_solution_tabu_search[i] = tabu_cost
    uncap_cost_solution_genetic_algorithm[i] = genetic_cost

    uncap_solution_solution_greedy[i] = greedy_solution
    uncap_solution_solution_local_search[i] = local_search_solution
    uncap_solution_solution_tabu_search[i] = tabu_solution
    uncap_solution_solution_genetic_algorithm[i] = genetic_solution

    print("========================================================================\n")