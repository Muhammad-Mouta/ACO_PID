from utils import Ant, Graph, initialize_ants, move_ant, calculate_cost, update_local_phermone, update_global_phermone, get_best_solution
from zn_pid_tuning import zn_pid_tune

def aco_pid_tune(g, n_ants, n_iterations, graph_size, alpha=0.5, beta=0.5, rho=0.5):
    """
    Tunes the parameters of a PID controller using Ant Colony Optimization algorithm.

    Parameters
    ----------
        g : control.TransferFunction.

        n_ants : int.

        n_iterations : int.

        graph_size : int.

        rho : float, = ]0, 1], default: 0.5.
            The evaporation rate.

    Returns
    -------
        tuple_of_float.
    """
    ants = initialize_ants(n_ants)

    # Create the graph
    k_p_mean, k_i_mean, k_d_mean = zn_pid_tune(g)
    graph = Graph(graph_size, k_p_mean, k_i_mean, k_d_mean)

    # Initialize the besst and the min_cost
    best_solution = {"p": k_p_mean, "i": k_i_mean, "d": k_d_mean}
    min_cost = calculate_cost(g, {"p": k_p_mean, "i": k_i_mean, "d": k_d_mean})
    print(best_solution)
    print(f"Total Cost: {min_cost[0]}")
    print(f"Maximum Peak Overshoot: {min_cost[1]}")
    print(f"Rise Time: {min_cost[2]}")
    print(f"Settling Time (2%): {min_cost[3]}")
    min_cost = min_cost[0]

    for i in range(n_iterations):
        # Initialize the best_ant
        best_ant = Ant()
        best_ant.cost = float("inf")

        for ant in ants:
            # Move the ant
            move_ant(ant, graph, alpha, beta)

            # Compute the path cost and update the best ant
            ant.cost = calculate_cost(g, ant.path["k"])[0]
            if ant.cost < best_ant.cost:
                best_ant = ant

            # Update the phermone locally
            update_local_phermone(ant, graph, min_cost)


        # Update the global phermone levels, apply evaporation and reinforce the path of the best ant
        update_global_phermone(best_ant, graph, rho, min_cost)

        # Update the min_cost and the best solution
        if best_ant.cost < min_cost:
            best_solution = best_ant.path["k"]
            print(best_solution)
            cost_vec = calculate_cost(g, best_ant.path["k"])
            print(f"Total Cost: {cost_vec[0]}")
            print(f"Maximum Peak Overshoot: {cost_vec[1]}")
            print(f"Rise Time: {cost_vec[2]}")
            print(f"Settling Time (2%): {cost_vec[3]}")
            min_cost = best_ant.cost

    return best_solution
