import pygad
import numpy as np
import random

def fitness_func(solution, solution_idx):
    output = solution[0]*function_inputs + solution[1]
    mse = np.mean((output - desired_output) ** 2)
    fitness = 1.0 / (mse + 1e-6)  
    return fitness

def callback_generation2(ga_instance):
    global last_fitness
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))
    print("Change     = {change}".format(change=(ga_instance.best_solution()[1] - last_fitness)))
    last_fitness = ga_instance.best_solution()[1]

function_inputs = np.linspace(-10,10,100)  # 100 puntos de entrada #x valor inicial random (random en la mente de Tomi)
desired_output = 5*function_inputs + 8 #A*x + b = x2

fitness_function = fitness_func

num_generations = 5000
num_parents_mating = 5

sol_per_pop = 50
num_genes = 2

init_range_low = -10
init_range_high = 10

parent_selection_type = "sss"
keep_parents = 5

crossover_type = "single_point"

mutation_type = "random"
mutation_percent_genes = 5
last_fitness = 0

ga_instance = pygad.GA(num_generations=num_generations,
                    num_parents_mating=num_parents_mating,
                    fitness_func=fitness_function,
                    sol_per_pop=sol_per_pop,
                    num_genes=num_genes,
                    init_range_low=init_range_low,
                    init_range_high=init_range_high,
                    parent_selection_type=parent_selection_type,
                    keep_parents=keep_parents,
                    crossover_type=crossover_type,
                    mutation_type=mutation_type,
                    mutation_percent_genes=mutation_percent_genes,
                    callback_generation=callback_generation2)

ga_instance.run()

ga_instance.plot_result()

solution, solution_fitness, solution_idx = ga_instance.best_solution()

print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

prediction = solution[0]*function_inputs + solution[1]
print("Predicted output based on the best solution : {prediction}".format(prediction=prediction))

if ga_instance.best_solution_generation != -1:
    print("Best fitness value reached after {best_solution_generation} generations.".format(best_solution_generation=ga_instance.best_solution_generation))