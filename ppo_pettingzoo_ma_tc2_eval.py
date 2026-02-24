import argparse
import joblib
import random
import signal
import torch
import traceback
from common.constants import AIRCRAFT_COUNT
from common.data_preprocessing import GNNProcessor
from envs.tc2_pettingzoo_env import make_env
from models.aircraft_agent import Agent, GNNAgent, ModelRegistry
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from utils.vec_envs import make_vec_env, SequentialVecEnv

exiting = False


def signal_handler(sig, frame):
    global exiting
    exiting = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--torch-deterministic", action=argparse.BooleanOptionalAction, default=True,
                        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=False,
                        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--num-steps", type=int, required=True,
                        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--model-path", type=str, default=None,
                        help="the path of the model to load")
    parser.add_argument("--agent-class", type=str, required=True,
                        help="agent class name (must match the saved model, if any)")
    parser.add_argument("--edge-criteria", type=str, choices=["fc", "dist_only", "dist_and_alt", "self_only"], required=True,
                        help="criteria to choose which nodes to connect edges between")
    parser.add_argument("--visualise-only", action=argparse.BooleanOptionalAction, default=False,
                        help="if true, will not automatically start simulator and will wait for user to manually start simulator for visualisation")
    parser.add_argument("--eval-episodes", type=int, default=256,
                        help="number of episodes to run the evaluation for; ignored if --visualise-only")
    args = parser.parse_args()
    return args


def _tensor_to_graph(obs: torch.Tensor, gnn_preprocessor: GNNProcessor, device: torch.device, num_envs: int = 1) -> Data:
    """Convert raw observation tensor to batched PyG Data for GNNAgent (single env in visualise)."""
    input_graphs = []
    for i in range(obs.shape[0]):
        input_graphs.append(gnn_preprocessor.preprocess_data(torch.Tensor(obs[i]))[0])
    return next(iter(DataLoader(input_graphs, batch_size=num_envs))).to(device)


if __name__ == "__main__":
    args = parse_args()
    print(args)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    num_envs = 1

    env_ids = ["0"] if args.visualise_only else [f"0_{random.randbytes(3).hex()}"]

    envs = make_vec_env(
        SequentialVecEnv, env_ids, make_env,
        ac_type_one_hot_encoder=joblib.load("common/recat_one_hot_encoder.joblib"),
        init_sim=not args.visualise_only, reset_print_period=50, max_steps=args.num_steps,
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

    agent.load_state_dict(torch.load(args.model_path)['agent'])
    print(f"Agent loaded from {args.model_path}")

    signal.signal(signal.SIGINT, signal_handler)
    try:
        # Metrics tracker
        reward_sum = 0
        lifespan_sum = 0
        episode_end_info = {
            'landing_rate': 0,
            'aircraft_conflict_rate': 0,
            'mva_conflict_rate': 0,
            'wake_conflict_rate': 0,
        }

        episode_no = 0
        with tqdm(total=args.eval_episodes, unit="eps") as pbar:
            while episode_no < args.eval_episodes or args.visualise_only:
                # Agent lifespan tracking
                masks = torch.zeros((args.num_steps, AIRCRAFT_COUNT)).to(device)

                # Start the game
                next_obs, _ = envs.reset()
                ac_mask = torch.IntTensor(next_obs[:,:,-1]).to(device)
                if is_gnn_agent:
                    next_obs = _tensor_to_graph(next_obs, gnn_preprocessor, device, num_envs)
                else:
                    next_obs = torch.Tensor(next_obs).to(device)

                for step in range(0, args.num_steps):
                    masks[step] = ac_mask

                    # ALGO LOGIC: action logic
                    with torch.no_grad():
                        # use_mode=True: select highest probability action (deterministic evaluation)
                        if is_gnn_agent:
                            action, _, _, _ = agent.get_action_and_value(next_obs, ac_mask, use_mode=True)
                            # GNNAgent returns action shape (num_envs, num_aircraft, action_dim) from _pad_actions
                        else:
                            action, _, _, _ = agent.get_action_and_value(next_obs[:,:,:-1], use_mode=True)
                            action = action.permute((1, 2, 0))  # Reshape action to (num_envs, aircraft_count, action_dim)

                        next_obs, reward, termination, truncation, infos = envs.step(
                            torch.cat((action, ac_mask.unsqueeze(-1)), dim=-1).cpu().numpy()
                        )
                        reward_sum += reward.sum().item()

                        ac_mask = torch.IntTensor(next_obs[:,:,-1]).to(device)
                        if is_gnn_agent:
                            next_obs = _tensor_to_graph(
                                torch.Tensor(next_obs), gnn_preprocessor, device, num_envs
                            )
                        else:
                            next_obs = torch.Tensor(next_obs).to(device)
                        next_termination = torch.Tensor(termination).to(device)
                        next_truncation = torch.Tensor(truncation).to(device)

                        next_active_agents = ac_mask - next_termination
                        terminate = (next_active_agents.sum(dim=-1) == 0 & next_termination.any(dim=-1)).squeeze().item()

                        if terminate or exiting:
                            break

                for key, value in infos[0][0].items():
                    episode_end_info[key] += value

                active_mask = masks.bool()
                agent_lifespans = torch.where(active_mask[-1,:], active_mask.shape[0], active_mask.sum(dim=0))
                avg_agent_lifespan = agent_lifespans.sum() / (agent_lifespans > 0).sum()  # Exclude agents that didn't appear at all
                lifespan_sum += avg_agent_lifespan.item()

                episode_no += 1
                pbar.update(1)

                if exiting:
                    print("Ctrl-C pressed")
                    break

        print(f"Average episode reward: {reward_sum / episode_no:.3f}")
        print(f"Average lifespan: {lifespan_sum / episode_no:.3f}")
        for key, value in episode_end_info.items():
            print(f"{key}: {value / episode_no:.5f}")
    except KeyboardInterrupt:
        print("Ctrl-C pressed")
    except:
        print(traceback.format_exc())
        print("Error encountered")
    finally:
        print("Exiting and cleaning up")
        envs.close()