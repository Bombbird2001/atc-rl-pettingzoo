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
    parser.add_argument("--seed", type=int, default=777,
                        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", action=argparse.BooleanOptionalAction, default=True,
                        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=False,
                        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--num-steps", type=int, required=True,
                        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--num-envs", type=int, required=True,
                        help="number of parallel environments for evaluation; defaults to 1 if --visualise-only")
    parser.add_argument("--model-path", type=str, default=None,
                        help="the path of the model to load (single model evaluation)")
    parser.add_argument("--model-folder", type=str, default=None,
                        help="folder to iterate agent_0.pt, agent_1.pt, ... and run evaluation for each; ignored if --model-path is set")
    parser.add_argument("--agent-class", type=str, required=True,
                        help="agent class name (must match the saved model, if any)")
    parser.add_argument("--edge-criteria", type=str, choices=["fc", "dist_only", "dist_and_alt", "self_only"], required=True,
                        help="criteria to choose which nodes to connect edges between")
    parser.add_argument("--visualise-only", action=argparse.BooleanOptionalAction, default=False,
                        help="if true, will not automatically start simulator and will wait for user to manually start simulator for visualisation")
    parser.add_argument("--eval-episodes", type=int, default=256,
                        help="number of episodes to run the evaluation for; ignored if --visualise-only")
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


def _discover_agent_checkpoints(folder: str):
    """Return list of (x, path) for agent_x.pt with x integer >= 0, sorted by x."""
    pattern = re.compile(r"^agent_(\d+)\.pt$")
    out = []
    for name in os.listdir(folder):
        m = pattern.match(name)
        if m:
            x = int(m.group(1))
            out.append((x, os.path.join(folder, name)))
    return sorted(out, key=lambda p: p[0])


def _tensor_to_graph(obs: torch.Tensor, gnn_preprocessor: GNNProcessor, device: torch.device, num_envs: int = 1) -> Data:
    """Convert raw observation tensor to batched PyG Data for GNNAgent (single env in visualise)."""
    input_graphs = []
    for i in range(obs.shape[0]):
        input_graphs.append(gnn_preprocessor.preprocess_data(torch.Tensor(obs[i]))[0])
    return next(iter(DataLoader(input_graphs, batch_size=num_envs))).to(device)


if __name__ == "__main__":
    args = parse_args()
    print(args)

    if args.model_path is not None:
        model_list = [(None, args.model_path)]
    elif args.model_folder is not None:
        model_list = _discover_agent_checkpoints(args.model_folder)
        if not model_list:
            raise FileNotFoundError(f"No agent_<x>.pt files found in {args.model_folder}")
    else:
        raise ValueError("One of --model-path or --model-folder is required")

    run = None
    if args.track:
        import wandb
        run = wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args),
            name=os.path.basename(args.model_folder or args.model_path or "eval"),
        )

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    num_envs = args.num_envs

    env_ids = [f"0_{random.randbytes(3).hex()}"] if args.visualise_only else [f"{i}_{random.randbytes(3).hex()}" for i in range(num_envs)]

    envs = make_vec_env(
        ParallelThreadVecEnv, env_ids, make_env,
        ac_type_one_hot_encoder=joblib.load("common/recat_one_hot_encoder.joblib"),
        init_sim=not args.visualise_only, reset_print_period=int(ceil(args.eval_episodes / args.num_envs)), max_steps=args.num_steps,
        is_eval=True, goal_reward=0, mva_penalty=0, conflict_penalty=0, wake_penalty=0
    )

    agent_type = ModelRegistry.get_model(args.agent_class)
    is_gnn_agent = ModelRegistry.model_is_gnn(args.agent_class)
    if is_gnn_agent:
        agent = GNNAgent(envs, 18, 2, agent_type).to(device)
        gnn_preprocessor = GNNProcessor(args.edge_criteria)
    else:
        agent = Agent(envs, agent_type).to(device)
        gnn_preprocessor = None

    signal.signal(signal.SIGINT, signal_handler)
    try:
        model_iter = tqdm(model_list, unit="model", disable=len(model_list) <= 1) if not args.visualise_only else model_list
        for step_x, model_path in model_iter:
            agent.load_state_dict(torch.load(model_path, map_location=device)["agent"])
            if step_x is not None:
                print(f"Evaluating agent_{step_x}.pt ({model_path})")
            else:
                print(f"Agent loaded from {model_path}")

            # Metrics tracker (per model)
            reward_sum = 0
            lifespan_sum = 0
            episode_end_info = {
                "landing_rate": 0,
                "aircraft_conflict_rate": 0,
                "mva_conflict_rate": 0,
                "wake_conflict_rate": 0,
            }
            episode_no = 0
            total_agents = 0

            with tqdm(total=args.eval_episodes, unit="eps") as pbar:
                while episode_no < args.eval_episodes or args.visualise_only:
                    if args.visualise_only:
                        reward_sum = 0
                        lifespan_sum = 0
                        episode_end_info = {
                            "landing_rate": 0,
                            "aircraft_conflict_rate": 0,
                            "mva_conflict_rate": 0,
                            "wake_conflict_rate": 0,
                        }
                        episode_no = 0
                        total_agents = 0

                    # Agent lifespan tracking; shape (num_steps, num_envs, AIRCRAFT_COUNT)
                    masks = torch.zeros((args.num_steps, num_envs, AIRCRAFT_COUNT)).to(device)

                    # Start the game
                    next_obs, _ = envs.reset(seed=args.seed)
                    ac_mask = torch.IntTensor(next_obs[:, :, -1]).to(device)
                    if is_gnn_agent:
                        next_obs = _tensor_to_graph(next_obs, gnn_preprocessor, device, num_envs)
                    else:
                        next_obs = torch.Tensor(next_obs).to(device)

                    for step in range(0, args.num_steps):
                        masks[step] = ac_mask

                        # ALGO LOGIC: action logic
                        with torch.no_grad():
                            if is_gnn_agent:
                                action, _, _, _ = agent.get_action_and_value(next_obs, ac_mask, use_mode=True)
                            else:
                                action, _, _, _ = agent.get_action_and_value(next_obs[:, :, :-1], use_mode=True)
                                action = action.permute((1, 2, 0))

                            next_obs, reward, termination, truncation, infos = envs.step(
                                torch.cat((action, ac_mask.unsqueeze(-1)), dim=-1).cpu().numpy()
                            )
                            reward_sum += reward.sum().item()

                            ac_mask = torch.IntTensor(next_obs[:, :, -1]).to(device)
                            if is_gnn_agent:
                                next_obs = _tensor_to_graph(
                                    torch.Tensor(next_obs), gnn_preprocessor, device, num_envs
                                )
                            else:
                                next_obs = torch.Tensor(next_obs).to(device)
                            next_termination = torch.Tensor(termination).to(device)
                            next_truncation = torch.Tensor(truncation).to(device)

                            next_active_agents = ac_mask - next_termination
                            terminating_envs = (next_active_agents.sum(dim=-1) == 0) & next_termination.any(dim=-1)
                            terminate = False
                            for env_idx in torch.where(terminating_envs)[0]:
                                envs.early_reset(env_idx.item(), args.seed)
                                for key, value in infos[env_idx.item()][0].items():
                                    episode_end_info[key] += value
                                if next_active_agents.sum().item() == 0:
                                    # All agents terminated, exit the step loop early
                                    terminate = True

                            if terminate or exiting:
                                break

                    # Aggregate episode_end_info from all envs (like training script)
                    for env_idx in torch.where(next_active_agents.sum(dim=-1) > 0)[0]:
                        for key, value in infos[env_idx.item()][0].items():
                            episode_end_info[key] += value

                    active_mask = masks.bool()  # (num_steps, num_envs, AIRCRAFT_COUNT)
                    agent_lifespans = torch.where(active_mask[-1, :, :], active_mask.shape[0], active_mask.sum(dim=0))
                    n_active = (agent_lifespans > 0).sum().item()
                    avg_agent_lifespan = agent_lifespans.sum() / n_active if n_active > 0 else torch.tensor(0.0, device=device)
                    lifespan_sum += avg_agent_lifespan.item()

                    total_agents += n_active
                    episode_no += num_envs
                    pbar.update(num_envs)

                    if args.visualise_only:
                        avg_reward = reward_sum / total_agents if total_agents > 0 else 0
                        print(f"Average episode reward: {avg_reward:.3f}")
                        print(f"Average lifespan: {lifespan_sum:.3f}")
                        for key, value in episode_end_info.items():
                            print(f"{key}: {value:.5f}")

                    if exiting:
                        print("Ctrl-C pressed")
                        break

            if not exiting and episode_no > 0:
                avg_reward = reward_sum / total_agents if total_agents > 0 else 0
                avg_lifespan = lifespan_sum / (episode_no // num_envs)  # iterations, each with one avg_lifespan
                print(f"Average episode reward: {avg_reward:.3f}")
                print(f"Average lifespan: {avg_lifespan:.3f}")
                for key, value in episode_end_info.items():
                    print(f"{key}: {value / episode_no:.5f}")

                if args.track and run is not None:
                    global_step = step_x if step_x is not None else 0
                    log_dict = {
                        "episode/average_agent_reward": avg_reward,
                        "episode/average_agent_lifespan": avg_lifespan,
                    }
                    for key, value in episode_end_info.items():
                        log_dict[f"metrics/{key}"] = value / episode_no
                    run.log(log_dict, step=global_step)

            if exiting:
                break
    except KeyboardInterrupt:
        print("Ctrl-C pressed")
    except:
        print(traceback.format_exc())
        print("Error encountered")
    finally:
        print("Exiting and cleaning up")
        envs.close()