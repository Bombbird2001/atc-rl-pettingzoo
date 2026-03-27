import argparse
import joblib
import os
import re
import random
import signal
import torch
import traceback
from common.constants import AIRCRAFT_COUNT
from common.data_preprocessing import GNNProcessor
from envs.tc2_gym_env import NODE_FEATURE_DIMENSION
from envs.tc2_pettingzoo_env import make_env
from math import ceil
from models.aircraft_agent import Agent, GNNAgent, ModelRegistry
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from utils.vec_envs import make_vec_env, ParallelThreadVecEnv

exiting = False


def signal_handler(sig, frame):
    global exiting
    exiting = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, required=True,
                        help="the name of this experiment")
    parser.add_argument("--random-spawn-chance", type=float, default=0,
                        help="the probability of spawning at random heading from airport")
    parser.add_argument("--seed", type=int, default=777,
                        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", action=argparse.BooleanOptionalAction, default=True,
                        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=False,
                        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--num-steps", type=int, required=True,
                        help="the number of steps to run in each environment per policy rollout; ignored if --endless-episode")
    parser.add_argument("--num-envs", type=int, required=True,
                        help="number of parallel environments for evaluation; defaults to 1 if --visualise-only")
    parser.add_argument("--model-path", type=str, default=None,
                        help="the path of the model to load (single model evaluation)")
    parser.add_argument("--model-folder", type=str, default=None,
                        help="folder to iterate agent_0.pt, agent_1.pt, ... and run evaluation for each; ignored if --model-path is set")
    parser.add_argument("--model-folder-start", type=int, default=1,
                        help="starts iterating through model-folder from agent_XX.pt, defaults to 1 (the first model); ignored if --model-path is set")
    parser.add_argument("--model-folder-end", type=int, default=None,
                        help="stop iterating through model-folder at agent_XX.pt (inclusive); ignored if --model-path is set")
    parser.add_argument("--agent-class", type=str, required=True,
                        help="agent class name (must match the saved model, if any)")
    parser.add_argument("--edge-criteria", type=str, choices=["fc", "dist_only", "dist_and_alt", "self_only"], required=True,
                        help="criteria to choose which nodes to connect edges between")
    parser.add_argument("--visualise-only", action=argparse.BooleanOptionalAction, default=False,
                        help="if true, will not automatically start simulator and will wait for user to manually start simulator for visualisation")
    parser.add_argument("--eval-episodes", type=int, default=256,
                        help="number of episodes to run the evaluation for; ignored if --visualise-only or --endless-episode")
    parser.add_argument("--endless-episode", action=argparse.BooleanOptionalAction, default=False,
                        help="if true, will run the episode continuously without truncation or termination")
    parser.add_argument("--track", action=argparse.BooleanOptionalAction, default=True,
                        help="if toggled, this evaluation will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default=None,
                        help="the wandb project name (when --track)")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="the wandb entity (when --track)")
    args = parser.parse_args()
    if args.num_envs is None and args.visualise_only:
        args.num_envs = 1
    return args


def _discover_agent_checkpoints(folder: str, start: int, end: int | None):
    """Return list of (x, path) for agent_x.pt with x integer >= 0, sorted by x."""
    pattern = re.compile(r"^agent_(\d+)\.pt$")
    out = []
    for name in os.listdir(folder):
        m = pattern.match(name)
        if m:
            x = int(m.group(1))
            if x < start or end is not None and x > end:
                continue
            out.append((x, os.path.join(folder, name)))
    return sorted(out, key=lambda p: p[0])


def _tensor_to_graph(obs: torch.Tensor, gnn_preprocessor: GNNProcessor, device: torch.device, num_envs: int = 1) -> Data:
    """Convert raw observation tensor to batched PyG Data for GNNAgent (single env in visualise)."""
    input_graphs = []
    for i in range(obs.shape[0]):
        input_graphs.append(gnn_preprocessor.preprocess_data(torch.Tensor(obs[i]))[0])
    return next(iter(DataLoader(input_graphs, batch_size=num_envs))).to(device)


def reset_episode_counters():
    return {
        "landing_rate": 0,
        "aircraft_conflict_rate_no_loc": 0,
        "mva_conflict_rate": 0,
        "wake_conflict_rate_no_loc": 0,
        "aircraft_conflict_rate_loc": 0,
        "wake_conflict_rate_loc": 0,
        "aircraft_conflict_rate": 0,
        "wake_conflict_rate": 0,
        'aircraft_conflict_rate_no_loc_before_res': 0,
        'aircraft_conflict_rate_loc_before_res': 0,
        'aircraft_conflict_rate_before_res': 0,
        'mva_conflict_rate_before_res': 0,
        'wake_conflict_rate_no_loc_before_res': 0,
        'wake_conflict_rate_loc_before_res': 0,
        'wake_conflict_rate_before_res': 0,
        'clearance_change_rate': 0,
    }


NODE_FEATURE_DIM = NODE_FEATURE_DIMENSION
EDGE_FEATURE_DIM = 2
RAW_STEP_EXTRA = 3900
SPAWN_GROUP_NAME_MAPPING = {
    0: "north",
    1: "east",
    2: "west-tabun",
    3: "west-sauna",
    4: "south"
}


if __name__ == "__main__":
    args = parse_args()
    print(args)

    if args.model_path is not None:
        model_list = [(None, args.model_path)]
    elif args.model_folder is not None:
        model_list = _discover_agent_checkpoints(args.model_folder, args.model_folder_start, args.model_folder_end)
        if not model_list:
            raise FileNotFoundError(f"No agent_<x>.pt files found in {args.model_folder} for start={args.model_folder_start}, end={args.model_folder_end}")
        print(model_list)
    else:
        raise ValueError("One of --model-path or --model-folder is required")

    run = None
    if args.track:
        import wandb
        run = wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args),
            name=f"{args.exp_name}_{os.path.basename(args.model_folder or args.model_path)}",
        )

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    num_envs = args.num_envs

    env_ids = [f"0_{random.randbytes(3).hex()}"] if args.visualise_only else [f"{i}_{random.randbytes(3).hex()}" for i in range(num_envs)]

    envs = make_vec_env(
        ParallelThreadVecEnv, env_ids, make_env,
        ac_type_one_hot_encoder=joblib.load("common/recat_one_hot_encoder_v2.joblib"),
        init_sim=not args.visualise_only, reset_print_period=int(ceil(args.eval_episodes / args.num_envs)),
        max_steps=args.num_steps if not args.endless_episode else None, is_eval=True, goal_reward=0, mva_penalty=0,
        conflict_penalty=0, wake_penalty=0, random_spawn_chance=args.random_spawn_chance, raw_step_extra=RAW_STEP_EXTRA
    )

    agent_type = ModelRegistry.get_model(args.agent_class)
    is_gnn_agent = ModelRegistry.model_is_gnn(args.agent_class)
    if is_gnn_agent:
        agent = GNNAgent(envs, NODE_FEATURE_DIM, EDGE_FEATURE_DIM, agent_type).to(device)
        gnn_preprocessor = GNNProcessor(args.edge_criteria)
    else:
        agent = Agent(envs, NODE_FEATURE_DIM, agent_type).to(device)
        gnn_preprocessor = None

    signal.signal(signal.SIGINT, signal_handler)
    log_step = 0
    try:
        model_iter = tqdm(model_list, unit="model", disable=len(model_list) <= 1) if not args.visualise_only else model_list
        for step_x, model_path in model_iter:
            agent.load_state_dict(torch.load(model_path, map_location=device)["agent"])
            if step_x is not None:
                print(f"Evaluating agent_{step_x}.pt ({model_path})")
            else:
                print(f"Agent loaded from {model_path}")

            # Metrics tracker (per model)
            reward_sum = 0.0
            lifespan_sum = 0
            episode_end_info = reset_episode_counters()
            aircraft_group_lifespans = dict()
            episode_no = 0
            times_added = 0
            total_agents = 0
            env_valid_steps = None
            raw_step = None

            with tqdm(total=args.eval_episodes, unit="eps") as pbar:
                while not exiting and (episode_no < args.eval_episodes or args.visualise_only or args.endless_episode):
                    spawn_groups = [[] for _ in range(num_envs)]
                    if args.visualise_only:
                        reward_sum = 0.0
                        lifespan_sum = 0
                        episode_end_info = reset_episode_counters()
                        episode_no = 0
                        times_added = 0
                        total_agents = 0

                    # Agent lifespan tracking; indexed by logical step (0..args.num_steps-1) per env
                    masks = torch.zeros((args.num_steps, num_envs, AIRCRAFT_COUNT)).to(device)

                    # Start the game
                    # print("Waiting reset")
                    # if env_valid_steps is not None:
                    #     print(f"Before reset - Raw step: {raw_step}, env steps: {env_valid_steps}")
                    next_obs, _ = envs.reset(seed=args.seed)
                    # print("Reset done")
                    ac_mask = torch.IntTensor(next_obs[:, :, -1]).to(device)
                    if is_gnn_agent:
                        next_obs = _tensor_to_graph(next_obs, gnn_preprocessor, device, num_envs)
                    else:
                        next_obs = torch.Tensor(next_obs).to(device)

                    # We may run "extra" internal env steps for conflict-resolution simulation.
                    # We index stats by logical step per env and control inclusion via masks.
                    rewards = torch.zeros((args.num_steps, num_envs, AIRCRAFT_COUNT), device=device)
                    # Per-(logical_step, env) validity mask for metrics (reward, etc.)
                    step_valid = torch.zeros((args.num_steps, num_envs), dtype=torch.bool, device=device)
                    # Per-environment count of valid logical steps recorded so far
                    env_valid_steps = torch.zeros(num_envs, dtype=torch.int, device=device)

                    raw_step = 0
                    raw_step_limit = args.num_steps + RAW_STEP_EXTRA  # allow extra sim steps; defensive cap
                    # Continue when there is at least one non-terminated env with < args.num_steps valid steps
                    terminated_envs = torch.zeros(num_envs, dtype=torch.bool, device=device)
                    while not exiting and ((((env_valid_steps < args.num_steps) & ~terminated_envs).any().item()) or args.endless_episode):
                        # ALGO LOGIC: action logic
                        if raw_step >= raw_step_limit and not args.endless_episode:
                            print(f"Warning: raw_step exceeded cap ({raw_step_limit}); breaking episode early with termination status {terminated_envs}")
                            break
                        with torch.no_grad():
                            if args.endless_episode and not ac_mask.any():
                                next_obs, _, _, _, infos = envs.step(
                                    torch.cat((
                                        torch.zeros(ac_mask.shape + envs.single_action_space.shape, dtype=torch.long), ac_mask.unsqueeze(-1)
                                    ), dim=-1).cpu().numpy()
                                )
                                ac_mask = torch.IntTensor(next_obs[:, :, -1]).to(device)
                                continue

                            if is_gnn_agent:
                                action, _, _, _ = agent.get_action_and_value(next_obs, ac_mask, use_mode=True)
                            else:
                                action, _, _, _ = agent.get_action_and_value(next_obs[:, :, :-2], use_mode=True)
                                action = action.permute((1, 2, 0))

                            next_obs, reward, termination, truncation, infos = envs.step(
                                torch.cat((action, ac_mask.unsqueeze(-1)), dim=-1).cpu().numpy()
                            )

                            # Handle step count offsets (for conflict avoidance simulation)
                            # Per-environment semantics:
                            # - new_step_count == 1: normal logical step for that env (record stats at current effective_step)
                            # - new_step_count == 0: sim step for that env (ignore for metrics)
                            # - new_step_count < 0: ignore the last |new_step_count| logical rows for that env in metrics
                            step_offsets = [info_dict[0].get("step_offset", 0) if 0 in info_dict else -1 for info_dict in infos]
                            new_step_counts = [1 + step_offset for step_offset in step_offsets]

                            # For environments with negative offset, invalidate last |new_step_count| logical steps for that env
                            for env_idx_i, nsc in enumerate(new_step_counts):
                                if nsc < 0:
                                    reset_len = abs(nsc)
                                    end = env_valid_steps[env_idx_i].item()
                                    start = max(end - reset_len, 0)
                                    if end > 0 and start < end:
                                        step_valid[start:end, env_idx_i] = False
                                        env_valid_steps[env_idx_i] = max(
                                            env_valid_steps[env_idx_i] - (end - start),
                                            torch.tensor(0, device=device),
                                        )

                            # If at least one environment has new_step_count == 1 AND still needs steps,
                            # we record stats for those envs at their current logical step index.
                            for env_idx_i, nsc in enumerate(new_step_counts):
                                if nsc == 1 and env_valid_steps[env_idx_i] < args.num_steps:
                                    step_idx = env_valid_steps[env_idx_i].item()
                                    masks[step_idx, env_idx_i] = ac_mask[env_idx_i]
                                    rewards[step_idx, env_idx_i] = torch.tensor(
                                        reward[env_idx_i], device=device
                                    )
                                    step_valid[step_idx, env_idx_i] = True
                                    env_valid_steps[env_idx_i] += 1

                            # if truncation.any():
                            #     print("Truncating at", env_valid_steps)

                            ac_mask = torch.IntTensor(next_obs[:, :, -1]).to(device)
                            if is_gnn_agent:
                                next_obs = _tensor_to_graph(
                                    torch.Tensor(next_obs), gnn_preprocessor, device, num_envs
                                )
                            else:
                                next_obs = torch.Tensor(next_obs).to(device)

                            if not args.endless_episode:
                                next_termination = torch.Tensor(termination).to(device)
                                next_truncation = torch.Tensor(truncation).to(device)
                                next_active_agents = ac_mask - next_termination
                                terminating_envs = (next_active_agents.sum(dim=-1) == 0) & next_termination.any(dim=-1)
                                terminate = False
                                for env_idx in torch.where(terminating_envs | ((env_valid_steps >= args.num_steps) & ~terminated_envs))[0]:
                                    # print("Early resetting env", env_idx.item())
                                    envs.early_reset(env_idx.item(), args.seed)
                                    terminated_envs[env_idx] = True
                                    times_added += 1
                                    for key, value in infos[env_idx.item()][0].items():
                                        if key == "step_offset":
                                            continue
                                        if key == "spawn_groups":
                                            spawn_groups[env_idx] = value
                                            continue
                                        episode_end_info[key] += value
                                    if next_active_agents.sum().item() == 0:
                                        # All agents terminated, exit the step loop early
                                        terminate = True

                                    if terminate:
                                        break

                        raw_step += 1

                    # Accumulate per-episode reward based only on valid (logical_step, env) entries
                    if step_valid.any():
                        step_env_rewards = rewards.sum(dim=-1)  # (steps, envs)
                        valid_rewards = step_env_rewards[step_valid]
                        reward_sum += valid_rewards.sum().item()

                    # Aggregate episode_end_info from all envs (like training script)
                    for env_idx in torch.where(~terminated_envs)[0]:
                        times_added += 1
                        for key, value in infos[env_idx.item()][0].items():
                            if key == "step_offset":
                                continue
                            if key == "spawn_groups":
                                spawn_groups[env_idx] = value
                                continue
                            episode_end_info[key] += value

                    # Lifespans based on valid (logical_step, env) entries
                    active_mask = masks.bool() & step_valid.unsqueeze(-1)  # (steps, num_envs, AIRCRAFT_COUNT)
                    # If an aircraft is still active at the final step, treat its lifespan as full horizon
                    agent_lifespans = torch.where(active_mask[-1, :, :], active_mask.shape[0], active_mask.sum(dim=0))
                    n_active = (agent_lifespans > 0).sum().item()
                    avg_agent_lifespan = agent_lifespans.sum() / n_active if n_active > 0 else torch.tensor(0.0, device=device)
                    lifespan_sum += avg_agent_lifespan.item()

                    for idx in range(num_envs):
                        for group, lifespan in zip(spawn_groups[idx], agent_lifespans[idx].tolist()):
                            if lifespan == 0 or lifespan == args.num_steps:
                                continue
                            if group not in aircraft_group_lifespans:
                                aircraft_group_lifespans[group] = []
                            aircraft_group_lifespans[group].append(lifespan)

                    total_agents += n_active
                    episode_no += num_envs
                    if episode_no != times_added:
                        print(f"Episodes: {episode_no}, times added: {times_added}")
                    pbar.update(num_envs)

                    if args.visualise_only:
                        avg_reward = (reward_sum / total_agents) if total_agents > 0 else 0
                        print(f"Average episode reward: {avg_reward:.3f}")
                        print(f"Average lifespan: {lifespan_sum:.3f}")
                        for key, value in episode_end_info.items():
                            print(f"{key}: {value:.5f}")

            if exiting:
                break

            if not exiting and episode_no > 0:
                print(f"Episodes: {episode_no}, times added: {times_added}")
                avg_reward = (reward_sum / total_agents) if total_agents > 0 else 0
                avg_lifespan = lifespan_sum / (episode_no // num_envs)  # iterations, each with one avg_lifespan
                print(f"Average episode reward: {avg_reward:.3f}")
                print(f"Average lifespan: {avg_lifespan:.3f}")
                for key, value in episode_end_info.items():
                    print(f"{key}: {value / episode_no:.5f}")

                if args.track and run is not None:
                    log_dict = {
                        "episode/average_agent_reward": avg_reward,
                        "episode/average_agent_lifespan": avg_lifespan,
                    }
                    for key, value in episode_end_info.items():
                        log_dict[f"metrics/{key}"] = value / episode_no

                    # Log aircraft lifespan standard deviation and distribution to histogram, grouped by spawn groups
                    data_table = []
                    for group, lifespans in aircraft_group_lifespans.items():
                        # tmp_table = wandb.Table(data=[[lifespan] for lifespan in lifespans], columns=["lifespan"])
                        # log_dict[f"agent_{step_x}/spawn-{SPAWN_GROUP_NAME_MAPPING[group]}-dist"] = wandb.plot.histogram(tmp_table, "lifespan", title=f"spawn-{SPAWN_GROUP_NAME_MAPPING[group]} Lifespans")
                        data_table.extend([[group, lifespan] for lifespan in lifespans])
                        log_dict[f"spawn/spawn-{SPAWN_GROUP_NAME_MAPPING[group]}-lifespan-dist"] = wandb.Histogram(lifespans)
                        log_dict[f"spawn/spawn-{SPAWN_GROUP_NAME_MAPPING[group]}-lifespan-std-dev"] = torch.FloatTensor(lifespans).std().item()
                    run.summary[f"agent_{step_x}/spawn-group-lifespan"] = wandb.Table(data=data_table, columns=["group", "lifespan"])
                    run.log(log_dict, step=log_step)

            log_step += 1
    except KeyboardInterrupt:
        print("Ctrl-C pressed")
    except:
        print(traceback.format_exc())
        print("Error encountered")
    finally:
        print("Exiting and cleaning up")
        envs.close()