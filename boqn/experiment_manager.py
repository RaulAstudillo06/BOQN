from typing import Callable, List, Optional

import os
import sys

from bofn_trial import bofn_trial
from dag import DAG


def experiment_manager(
    first_trial: int, 
    last_trial: int,
    problem: str,
    function_network: Callable,
    network_to_objective_transform: Callable,
    input_dim: int,
    dag: DAG,
    active_input_indices: List[List[Optional[int]]],
    algos: List[str],
    n_init_evals: int,
    n_bo_iter: int,
    restart: bool,
) -> None:
    script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
    project_path = script_dir[:-5]

    for algo in algos:
        results_folder = project_path + "/experiments_results/" + problem + "/" + algo + "/"
        if not os.path.exists(results_folder) :
            os.makedirs(results_folder)
        if not os.path.exists(results_folder + "running_times/"):
            os.makedirs(results_folder + "running_times/")
        if not os.path.exists(results_folder + "X/"):
            os.makedirs(results_folder + "X/")
        if not os.path.exists(results_folder + "network_output_at_X/"):
            os.makedirs(results_folder + "network_output_at_X/")
        if not os.path.exists(results_folder + "objective_at_X/"):
            os.makedirs(results_folder + "objective_at_X/")

        for trial in range(first_trial, last_trial + 1):
            bofn_trial(
                problem=problem,
                function_network=function_network,
                network_to_objective_transform=network_to_objective_transform,
                input_dim=input_dim,
                dag=dag,
                active_input_indices=active_input_indices,
                algo=algo,
                n_init_evals=n_init_evals,
                n_bo_iter=n_bo_iter,
                trial=trial,
                restart=restart,
            )
            