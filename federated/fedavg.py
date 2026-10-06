import numpy as np


def fedavg(client_parameters, client_sizes):
    """
    Perform weighted Federated Averaging.

    Parameters:
        client_parameters:
            List of parameter lists from each client.

        client_sizes:
            Number of training samples for each client.

    Returns:
        Global averaged parameters.
    """

    total_samples = sum(client_sizes)

    num_parameter_arrays = len(client_parameters[0])

    global_parameters = []

    for parameter_index in range(num_parameter_arrays):

        weighted_sum = None

        for client_index in range(len(client_parameters)):

            client_parameter = client_parameters[client_index][parameter_index]

            client_weight = (
                client_sizes[client_index] / total_samples
            )

            if weighted_sum is None:
                weighted_sum = (
                    client_weight * client_parameter
                )
            else:
                weighted_sum += (
                    client_weight * client_parameter
                )

        global_parameters.append(weighted_sum)

    return global_parameters