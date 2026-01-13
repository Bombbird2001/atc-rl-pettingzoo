import argparse
import joblib
import signal
import torch
import traceback
from envs.tc2_pettingzoo_env import make_env
from models.aircraft_agent import MLPAgent
from utils.vec_envs import SequentialVecEnv


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
    parser.add_argument("--num-steps", type=int, default=256,
                        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--model-path", type=str, default=None,
                        help="the path of the model to load")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = SequentialVecEnv.make_vec_env(
        1, make_env,
        ac_type_one_hot_encoder=joblib.load("common/recat_one_hot_encoder.joblib"),
        init_sim=False, reset_print_period=50, max_steps=args.num_steps
    )

    agent = MLPAgent(envs).to(device)
    agent.load_state_dict(torch.load(args.model_path))
    print(f"Agent loaded from {args.model_path}")

    signal.signal(signal.SIGINT, signal_handler)
    try:
        while True:
            # Start the game
            next_obs, _ = envs.reset()
            next_obs = torch.Tensor(next_obs).to(device)
            reward_sum = 0

            for step in range(0, args.num_steps):
                # ALGO LOGIC: action logic
                with torch.no_grad():
                    # use_mode=True, always select the highest probability action instead of sampling (for training)
                    action, _, _, _ = agent.get_action_and_value(next_obs[:,:,:-1], use_mode=True)
                    action = action.permute((1, 2, 0))  # Reshape action to (num_envs, aircraft_count, action_dim)

                    next_obs, reward, termination, truncation, _ = envs.step(
                        # Concat the aircraft mask, ignores actions generated for non-existent aircraft entries
                        torch.cat((action, next_obs[:,:,-1].to(torch.int32).unsqueeze(-1)), dim=-1).cpu().numpy()
                    )
                    reward_sum += reward.sum().item()

                    next_obs, next_termination, next_truncation = (
                        torch.Tensor(next_obs).to(device),
                        torch.Tensor(termination).to(device),
                        torch.Tensor(truncation).to(device),
                    )

                    next_active_agents = next_obs[:,:,-1] - next_termination
                    terminate = (next_active_agents.sum(dim=-1) == 0 & next_termination.any(dim=-1)).squeeze().item()

                    if terminate or exiting:
                        break

            if exiting:
                print("Ctrl-C pressed, exiting")
                break

            print("Episode reward:", reward_sum)
    except KeyboardInterrupt:
        print("Ctrl-C pressed, exiting")
    except:
        print(traceback.format_exc())
        print("Error encountered, exiting")