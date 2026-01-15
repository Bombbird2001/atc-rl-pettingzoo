"""Advanced training script adapted from CleanRL's repository: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_pettingzoo_ma_atari.py.

This is a full training script including CLI, logging and integration with TensorBoard and WandB for experiment tracking.

Full documentation and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_pettingzoo_ma_ataripy.

Note: default value for total-timesteps has been changed from 2 million to 8000, for easier testing.

Authors: Costa (https://github.com/vwxyzjn), Elliot (https://github.com/elliottower)
"""

import argparse
import gymnasium as gym
import joblib
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import traceback
from collections import deque
from common.constants import AIRCRAFT_COUNT
from datetime import datetime
from envs.tc2_pettingzoo_env import make_env
from math import ceil
from models.aircraft_agent import MLPAgent
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils.buffers import RolloutBuffer
from utils.vec_envs import make_vec_env, ParallelThreadVecEnv


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=777,
                        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", action=argparse.BooleanOptionalAction, default=True,
                        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=False,
                        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", action=argparse.BooleanOptionalAction, default=True,
                        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default=None,
                        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="the entity (team) of wandb's project")
    # parser.add_argument("--capture-video", action=argparse.BooleanOptionalAction, default=False,
    #                     help="whether to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--auto-init-sim", action=argparse.BooleanOptionalAction, default=True,
                        help="if toggled, will automatically initialize the simulators for the environment")
    parser.add_argument("--model-path", type=str, default=None,
                        help="the path of the model to load (continue training from)")

    # Algorithm specific arguments
    parser.add_argument("--total-timesteps", type=int, default=12000,  # CleanRL default: 2000000
                        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
                        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=16,
                        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=256,
                        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--max-agents", type=int, default=None,
                        help="the maximum number of agents that can be present in game (this affects only the rollout buffer size)")
    parser.add_argument("--anneal-lr", action=argparse.BooleanOptionalAction, default=True,
                        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="the lambda for the general advantage estimation")
    parser.add_argument("--minibatch-size", type=int, default=32,
                        help="the size of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
                        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", action=argparse.BooleanOptionalAction, default=True,
                        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.1,
                        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", action=argparse.BooleanOptionalAction, default=True,
                        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
                        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.1,
                        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
                        help="the target KL divergence threshold")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps * args.max_agents)
    # fmt: on
    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)
    run_name = f"{args.exp_name}__{args.seed}__{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )

    # TRY NOT TO MODIFY: seeding
    # random.seed(args.seed)  # Disable since random is being used to generate random identifiers for shared memory
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = make_vec_env(
        ParallelThreadVecEnv,
        args.num_envs, make_env,
        ac_type_one_hot_encoder=joblib.load("common/recat_one_hot_encoder.joblib"),
        init_sim=args.auto_init_sim, reset_print_period=100, max_steps=args.num_steps
    )

    try:
        # if args.capture_video:
        #     envs = gym.wrappers.RecordVideo(envs, f"videos/{run_name}")
        assert isinstance(
            envs.single_action_space, gym.spaces.MultiDiscrete
        ), "only multi-discrete action space is supported"

        agent = MLPAgent(envs).to(device)
        if args.model_path is not None:
            agent.load_state_dict(torch.load(args.model_path))
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

        global_step = 0
        start_time = time.time()
        update = 0
        num_updates = int(ceil(args.total_timesteps / args.batch_size))

        rollout_buffer = RolloutBuffer(args.batch_size, device)
        reward_history_length = 30
        reward_history = deque()
        reward_history_sum = 0
        reward_history_squared_sum = 0

        with tqdm(total=args.total_timesteps, unit="steps") as pbar:
            while True:
                # Start the game
                next_obs, info = envs.reset(seed=args.seed)
                next_obs = torch.Tensor(next_obs).to(device)
                next_termination = torch.zeros(args.num_envs, AIRCRAFT_COUNT).to(device)
                next_truncation = torch.zeros(args.num_envs, AIRCRAFT_COUNT).to(device)

                # ALGO Logic: Storage setup
                obs = torch.zeros(
                    (args.num_steps, args.num_envs, AIRCRAFT_COUNT) + envs.single_observation_space.shape
                ).to(device)
                actions = torch.zeros(
                    (args.num_steps, args.num_envs, AIRCRAFT_COUNT) + envs.single_action_space.shape
                ).to(device)
                logprobs = torch.zeros((args.num_steps, args.num_envs, AIRCRAFT_COUNT)).to(device)
                rewards = torch.zeros((args.num_steps, args.num_envs, AIRCRAFT_COUNT)).to(device)
                terminations = torch.zeros((args.num_steps, args.num_envs, AIRCRAFT_COUNT)).to(device)
                truncations = torch.zeros((args.num_steps, args.num_envs, AIRCRAFT_COUNT)).to(device)
                values = torch.zeros((args.num_steps, args.num_envs, AIRCRAFT_COUNT)).to(device)

                for step in range(0, args.num_steps):
                    # obs[step] stores the observation observed at that step
                    # termination/truncation[step] is only 1 on the last step the agent appears in
                    # But rewards/terminations/truncations[step] stores the reward at step+1 after taking an action during step,
                    # and whether the subsequent obs is the last for that agent
                    obs[step] = next_obs
                    terminations[step] = next_termination
                    truncations[step] = next_truncation

                    # ALGO LOGIC: action logic
                    with torch.no_grad():
                        action, logprob, _, value = agent.get_action_and_value(next_obs[:,:,:-1])
                        values[step] = value.squeeze(dim=-1)
                    action = action.permute((1, 2, 0))  # Reshape action to (num_envs, aircraft_count, action_dim)
                    actions[step] = action
                    logprobs[step] = logprob

                    # TRY NOT TO MODIFY: execute the game and log data.
                    next_obs, reward, termination, truncation, info = envs.step(
                        # Concat the aircraft mask, ignores actions generated for non-existent aircraft entries
                        torch.cat((action, next_obs[:,:,-1].to(torch.int32).unsqueeze(-1)), dim=-1).cpu().numpy()
                    )
                    reward = reward.astype(np.float32)

                    rewards[step] = torch.tensor(reward).to(device)
                    next_obs, next_termination, next_truncation = (
                        torch.Tensor(next_obs).to(device),
                        torch.Tensor(termination).to(device),
                        torch.Tensor(truncation).to(device),
                    )

                    # Early termination if all agents terminated before truncation occurs
                    next_active_agents = next_obs[:,:,-1] - next_termination
                    # Only terminate envs that have no more agents and termination occurred this step
                    # (i.e. envs that terminated before do not re-terminate)
                    terminating_envs = (next_active_agents.sum(dim=-1) == 0) & next_termination.any(dim=-1)
                    for env_idx in torch.where(terminating_envs)[0]:
                        envs.early_reset(env_idx.item(), args.seed)

                with torch.no_grad():
                    # Here, our inputs are such that every agent will only have a single continuous active period between spawn and despawn/truncation
                    # But generally every agent does not spawn at the same time, nor despawn at the same time
                    # A number of cases:
                    # 1. Current step is active, and next step is also active
                    #    Use the GAE formula and next step reward information
                    # 2. Current step is active, and terminates (NOT truncates)
                    #    nextnonterminal becomes 0, next_value should simply use the final step reward and ignore bootstrapped values
                    # 3. Current step is active, and truncates (NOT terminates) - by definition this only occurs on the final step
                    #    nextnonterminal becomes 0, causing the GAE formula to use the bootstrapped critic value using the next (final) obs
                    # 4. Current step is inactive
                    #    Doesn't really matter how we handle this, since this and the previous steps should all be masked out during training

                    # This is only used for truncations, NOT terminations; use the final next_obs with mask removed
                    truncation_next_value = agent.get_value(next_obs[:,:,:-1]).squeeze(dim=-1)
                    advantages = torch.zeros_like(rewards).to(device)
                    lastgaelam = 0
                    # next_done = torch.maximum(next_termination, next_truncation)
                    # dones = torch.maximum(terminations, truncations)
                    for t in reversed(range(args.num_steps)):
                        if t == args.num_steps - 1:
                            # Distinguish between truncations, and agents who terminate on the final step
                            # Only truncations are required to add the bootstrapped next value
                            nextnonterminal = 1.0 - next_termination
                            nextvalues = truncation_next_value
                        else:
                            # If agent terminates, do not add next value
                            nextnonterminal = 1.0 - terminations[t + 1]
                            nextvalues = values[t + 1]
                        delta = (
                                rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                        )
                        advantages[t] = lastgaelam = (
                                delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                        )
                    returns = advantages + values

                # Flatten the batch and add to rollout buffer with fixed maximum size
                b_obs = obs.reshape((-1, obs.shape[3]))
                b_masks = b_obs[:,-1].to(torch.bool)
                b_obs = b_obs[b_masks,:-1]
                b_logprobs = logprobs.reshape(-1)[b_masks]
                b_actions = actions.reshape((-1,) + envs.single_action_space.shape)[b_masks]
                b_advantages = advantages.reshape(-1)[b_masks]
                b_returns = returns.reshape(-1)[b_masks]
                b_values = values.reshape(-1)[b_masks]

                # print(b_masks.sum(), b_obs.shape, b_logprobs.shape, b_actions.shape, b_advantages.shape, b_returns.shape, b_values.shape)
                rollout_buffer.add_data(b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values)

                if rollout_buffer.full:
                    buffer_obs, buffer_logprobs, buffer_actions, buffer_advantages, buffer_returns, buffer_values = rollout_buffer.get_data()

                    # Annealing the rate if instructed to do so.
                    if args.anneal_lr:
                        frac = 1.0 - (update - 1.0) / num_updates
                        lrnow = frac * args.learning_rate
                        optimizer.param_groups[0]["lr"] = lrnow

                    # Optimizing the policy and value network
                    b_inds = np.arange(args.batch_size)
                    clipfracs = []
                    for epoch in range(args.update_epochs):
                        np.random.shuffle(b_inds)
                        for start in range(0, args.batch_size, args.minibatch_size):
                            end = start + args.minibatch_size
                            mb_inds = b_inds[start:end]

                            _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                                buffer_obs[mb_inds], buffer_actions.long()[mb_inds].transpose(0, 1)
                            )
                            logratio = newlogprob - buffer_logprobs[mb_inds]
                            ratio = logratio.exp()

                            with torch.no_grad():
                                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                                old_approx_kl = (-logratio).mean()
                                approx_kl = ((ratio - 1) - logratio).mean()
                                clipfracs += [
                                    ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                                ]

                            mb_advantages = buffer_advantages[mb_inds]
                            if args.norm_adv:
                                mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                                        mb_advantages.std() + 1e-8
                                )

                            # Policy loss
                            pg_loss1 = -mb_advantages * ratio
                            pg_loss2 = -mb_advantages * torch.clamp(
                                ratio, 1 - args.clip_coef, 1 + args.clip_coef
                            )
                            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                            # Value loss
                            newvalue = newvalue.view(-1)
                            if args.clip_vloss:
                                v_loss_unclipped = (newvalue - buffer_returns[mb_inds]) ** 2
                                v_clipped = buffer_values[mb_inds] + torch.clamp(
                                    newvalue - buffer_values[mb_inds],
                                    -args.clip_coef,
                                    args.clip_coef,
                                    )
                                v_loss_clipped = (v_clipped - buffer_returns[mb_inds]) ** 2
                                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                                v_loss = 0.5 * v_loss_max.mean()
                            else:
                                v_loss = 0.5 * ((newvalue - buffer_returns[mb_inds]) ** 2).mean()

                            entropy_loss = entropy.mean()
                            loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                            optimizer.zero_grad()
                            loss.backward()
                            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                            optimizer.step()

                        if args.target_kl is not None:
                            if approx_kl > args.target_kl:
                                break

                    y_pred, y_true = buffer_values.cpu().numpy(), buffer_returns.cpu().numpy()
                    var_y = np.var(y_true)
                    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

                    global_step += rollout_buffer.size
                    pbar.update(rollout_buffer.size)
                    update += 1
                    rollout_buffer.reset()

                    # TRY NOT TO MODIFY: record rewards for plotting purposes
                    writer.add_scalar(
                        "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
                    )
                    writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
                    writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
                    writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
                    writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
                    writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
                    writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
                    writer.add_scalar("losses/explained_variance", explained_var, global_step)
                    if update % 10 == 0:
                        print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar(
                        "charts/SPS", int(global_step / (time.time() - start_time)), global_step
                    )

                    # Use the rewards obtained from this iteration before training to compute average reward
                    ac_mask = obs.reshape(args.num_steps, -1, envs.single_observation_space.shape[0])[:, :,-1].squeeze(dim=-1).to(torch.bool).any(dim=0)
                    rewards = rewards.reshape(args.num_steps, -1)[:,ac_mask]
                    avg_agent_reward = rewards.sum(dim=0).mean()
                    if update % 10 == 0:
                        print("Average reward:", avg_agent_reward.item())
                    writer.add_scalar(
                        "episode/average_agent_reward", avg_agent_reward.item(), global_step
                    )

                    # SMA of reward
                    reward_history.append(avg_agent_reward.item())
                    reward_history_sum += avg_agent_reward.item()
                    reward_history_squared_sum += avg_agent_reward.item() ** 2
                    if len(reward_history) > reward_history_length:
                        to_remove = reward_history.popleft()
                        reward_history_sum -= to_remove
                        reward_history_squared_sum -= to_remove ** 2
                    writer.add_scalar(
                        "episode/reward_sma", reward_history_sum / len(reward_history), global_step
                    )
                    writer.add_scalar(
                        "episode/reward_variance", reward_history_squared_sum / len(reward_history) - (reward_history_sum / len(reward_history)) ** 2, global_step
                    )

                    # print(update, "of", num_updates)
                    if update >= num_updates:
                        break
        model_path = f"runs/{run_name}/agent.pt"
        print(f"Finished training in {time.time() - start_time:.2f}s")
        print(f"Saving model to {model_path}")
        torch.save(agent.state_dict(), model_path)
    except Exception as e:
        print(traceback.format_exc())
        print("Error encountered during training")
    finally:
        print("Exiting and cleaning up")
        envs.close()
        writer.close()