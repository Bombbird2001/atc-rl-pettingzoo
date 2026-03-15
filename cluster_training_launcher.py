import os
import platform
import psutil
import random
import subprocess
import sys
import time
from math import ceil
from typing import List


ENVS_PER_CPU = 2
SIMULATOR_JAR = os.getenv("SIMULATOR_JAR")


def get_arg(arg_name: str, input_args: List[str]) -> str:
    envs_flag = list(filter(lambda x: arg_name == x[1], enumerate(input_args)))
    if len(envs_flag) == 0:
        raise Exception(f"Expected {arg_name} flag")
    if len(envs_flag) > 1:
        raise Exception(f"Expected only a single {arg_name} flag")

    return input_args[envs_flag[0][0] + 1]


if __name__ == "__main__":
    p = psutil.Process()
    assigned_cores = None
    if platform.system() in ["Linux", "Windows", "FreeBSD"]:
        assigned_cores = p.cpu_affinity()
        print("Assigned cores:", assigned_cores)
    else:
        print("Core pinning not supported on", platform.system())

    script_to_invoke = sys.argv[1]
    script_args = sys.argv[2:]

    envs_to_start = int(get_arg("--num-envs", script_args))
    if assigned_cores is not None:
        required_cores = ceil(envs_to_start / ENVS_PER_CPU)
        if required_cores > len(assigned_cores):
            raise Exception(f"Requires {required_cores} cores for {envs_to_start} envs")

    # Get reward penalties to pass to simulator processes
    goal_reward = float(get_arg("--goal-reward", script_args))
    mva_penalty = float(get_arg("--mva-penalty", script_args))
    conflict_penalty = float(get_arg("--conflict-penalty", script_args))
    wake_penalty = float(get_arg("--wake-penalty", script_args))
    random_spawn_chance = float(get_arg("--random-spawn-chance", script_args))

    # Start trainer process first, pin to first core if possible
    env_ids = [f"{env_no}_{random.randbytes(3).hex()}" for env_no in range(envs_to_start)]
    script_args.append("--no-auto-init-sim")
    script_args.append("--env-ids")
    script_args.extend(env_ids)

    popen_args = []
    if assigned_cores:
        popen_args.extend(["taskset", "-c", str(assigned_cores[0])])
    popen_args.extend(["python", script_to_invoke])
    popen_args.extend(script_args)
    print("Launching", " ".join(popen_args))
    train_process = subprocess.Popen(popen_args)

    # Terrible way to wait for the script to start
    time.sleep(30)

    # Start all simulator processes
    sim_args = ["java", "-jar", SIMULATOR_JAR]
    env_processes = []
    for idx, env_id in enumerate(env_ids):
        taskset_args = []
        if assigned_cores:
            taskset_args.extend(["taskset", "-c", str(assigned_cores[(idx // ENVS_PER_CPU) + 1])])
        all_args = (
                taskset_args + sim_args +
                [
                    env_id, "0", str(goal_reward), str(mva_penalty), str(conflict_penalty),
                    str(wake_penalty), str(random_spawn_chance)
                ]
        )
        print("Launching", " ".join(all_args))
        env_processes.append(subprocess.Popen(all_args))

    train_process.wait()
    for env_process in env_processes:
        env_process.kill()
