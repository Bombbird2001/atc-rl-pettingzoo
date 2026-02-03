import argparse
import joblib
import signal
import torch
import traceback
from envs.tc2_pettingzoo_env import make_env
from models.aircraft_agent import MLPAgent, GNNAgent
from common.data_preprocessing import GNNProcessor
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
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
    parser.add_argument("--num-steps", type=int, default=512,
                        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--model-path", type=str, default=None,
                        help="the path of the model to load")
    parser.add_argument("--agent-type", type=str, default="gnn", choices=("mlp", "gnn"),
                        help="agent type: mlp or gnn (must match the saved model)")
    args = parser.parse_args()
    return args


def _tensor_to_graph(obs: torch.Tensor, gnn_preprocessor: GNNProcessor, device: torch.device, num_envs: int = 1) -> Data:
    """Convert raw observation tensor to batched PyG Data for GNNAgent (single env in visualise)."""
    input_graphs = []
    for i in range(obs.shape[0]):
        input_graphs.append(gnn_preprocessor.preprocess_data(torch.Tensor(obs[i])))
    return next(iter(DataLoader(input_graphs, batch_size=num_envs))).to(device)


if __name__ == "__main__":
    args = parse_args()
    print(args)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    num_envs = 1

    envs = make_vec_env(
        SequentialVecEnv,
        num_envs, make_env,
        ac_type_one_hot_encoder=joblib.load("common/recat_one_hot_encoder.joblib"),
        init_sim=False, reset_print_period=50, max_steps=args.num_steps,
        is_eval=True,
    )

    is_gnn_agent = args.agent_type == "gnn"
    if is_gnn_agent:
        agent = GNNAgent(envs, 18, 2).to(device)
        gnn_preprocessor = GNNProcessor()
    else:
        agent = MLPAgent(envs).to(device)
        gnn_preprocessor = None

    agent.load_state_dict(torch.load(args.model_path))
    print(f"Agent loaded from {args.model_path}")

    signal.signal(signal.SIGINT, signal_handler)
    try:
        while True:
            # Start the game
            next_obs, _ = envs.reset()
            ac_mask = torch.IntTensor(next_obs[:,:,-1]).to(device)
            if is_gnn_agent:
                next_obs = _tensor_to_graph(next_obs, gnn_preprocessor, device, num_envs)
            else:
                next_obs = torch.Tensor(next_obs).to(device)
            reward_sum = 0

            for step in range(0, args.num_steps):
                # ALGO LOGIC: action logic
                with torch.no_grad():
                    # use_mode=True: select highest probability action (deterministic evaluation)
                    if is_gnn_agent:
                        action, _, _, _ = agent.get_action_and_value(next_obs, ac_mask, use_mode=True)
                        # GNNAgent returns action shape (num_envs, num_aircraft, action_dim) from _pad_actions
                    else:
                        action, _, _, _ = agent.get_action_and_value(next_obs[:,:,:-1], use_mode=True)
                        action = action.permute((1, 2, 0))  # Reshape action to (num_envs, aircraft_count, action_dim)

                    next_obs, reward, termination, truncation, _ = envs.step(
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

            if exiting:
                print("Ctrl-C pressed")
                break

            print("Episode reward:", reward_sum)
    except KeyboardInterrupt:
        print("Ctrl-C pressed")
    except:
        print(traceback.format_exc())
        print("Error encountered")
    finally:
        print("Exiting and cleaning up")
        envs.close()