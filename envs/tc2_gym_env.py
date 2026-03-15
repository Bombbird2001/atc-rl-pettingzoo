import gymnasium as gym
import numpy as np
import os
import pandas as pd
import platform
import signal
import subprocess
import torch
from common.constants import AIRCRAFT_COUNT, SPD_BIAS, SPD_SCALE_DOWN, \
    TRACK_RATE_SCALE_DOWN, X_Y_SCALE_DOWN, PX_PER_NM, ALT_SCALE_DOWN, \
    ALT_RATE_SCALE_DOWN, ALT_BIAS, APP_SPD_BIAS, APP_SPD_SCALE_DOWN
from common.data_preprocessing import RECAT_MAPPING
from gymnasium import spaces
from sklearn.preprocessing import OneHotEncoder
from utils.game_bridge import GameBridge


OBS_SPACE_DIMENSION = 21
NODE_FEATURE_DIMENSION = OBS_SPACE_DIMENSION - 2
SIMULATOR_JAR = os.getenv("SIMULATOR_JAR")


class TC2GymEnv(gym.Env):
    def __init__(
            self, ac_type_one_hot_encoder: OneHotEncoder, goal_reward: float, mva_penalty: float,
            conflict_penalty: float, wake_penalty: float, random_spawn_chance: float, is_eval=False,
            render_mode=None, reset_print_period=50, env_id="", init_sim=True, max_steps=300,
            raw_step_extra=0,
    ):
        super().__init__()

        self.init_sim = init_sim
        self.sim_bridge = GameBridge.get_bridge_for_platform(instance_suffix=env_id)
        self.signalled_ready = False

        self.instance_name = f"env{env_id}"

        self.is_eval = is_eval
        self.reset_print_period = reset_print_period

        # Actions[0] = [-45, -20, -10, 0, +10, +20, +45 degrees]
        # Actions[1] = [-3000, -1000, 0, +1000]
        # Actions[2] = [-30, -10, 0, +10 knots]
        # self.action_space = spaces.MultiDiscrete([7, 4, 4])
        self.action_space = spaces.MultiDiscrete([5, 5, 5])

        # [aircraft type, x, y, alt, gs, track, angular speed, vertical speed,
        # current cleared altitude, current cleared heading, current cleared speed, localizer captured] normalized
        # +1 for aircraft masking
        # +1 for action masking
        self.OBS_SPACE_DIMENSION = OBS_SPACE_DIMENSION
        self.observation_space = spaces.Box(
            low=np.repeat(-1.0, self.OBS_SPACE_DIMENSION),
            high=np.repeat(1.0, self.OBS_SPACE_DIMENSION),
            dtype=np.float32
        )

        # Remove nuisance missing feature name warning since we're using numpy during inference with no column names
        ac_type_one_hot_encoder.feature_names_in_ = None
        self.ac_type_one_hot_encoder = ac_type_one_hot_encoder

        self.episode = 0
        self.steps = 0
        self.max_steps = max_steps
        self.raw_steps = 0
        self.max_raw_steps = max_steps + raw_step_extra
        self.terminated_count = 0
        self.render_mode = render_mode

        self.action_dist = []

        print(f"[{self.instance_name}] Environment initialized")

        if init_sim:
            print(f"[{self.instance_name}] Starting simulator")
            env_args = [
                "java", "-jar", SIMULATOR_JAR, env_id, "1" if is_eval else "0",
                str(goal_reward), str(mva_penalty), str(conflict_penalty), str(wake_penalty),
                str(random_spawn_chance)
            ]
            self.sim_process = subprocess.Popen(env_args)
        else:
            print(f"[{self.instance_name}] init_sim is False, simulator will not start automatically")

    def _get_observation_from_aircraft_state(self, aircraft_state) -> np.ndarray:
        tmp_state = np.array(aircraft_state).reshape(AIRCRAFT_COUNT, -1)[:,1:]
        # print(tmp_state)
        ac_types = np.array(tmp_state[:,:4], dtype=np.str_)
        ac_types = [[RECAT_MAPPING.get(ac_type, "Unknown")] for ac_type in (ac_types[:,0] + ac_types[:,1] + ac_types[:,2] + ac_types[:,3])]
        ac_type_one_hot = self.ac_type_one_hot_encoder.transform(ac_types).toarray()
        # Map
        # ICAO type, x, y, alt, ias, track, track rate, vertical speed, cleared alt, cleared hdg, cleared IAS, approach IAS, LOC cap, mask, terminated, action mask, agent ID
        # to
        # ["ias", "track_rate", "x", "y", "combined_alt", "combined_alt_rate", "track_x", "track_y", "prev_cleared_hdg_x", "prev_cleared_hdg_y",
        # "prev_cleared_alt", "prev_cleared_ias", "approach speed"] + aircraft_recat_one_hot
        # + ["action mask", (ONLY ADD NEW ITEMS BEFORE MASK AND AGENT_ID) "mask", "agent ID"]
        ac_state = np.array(tmp_state[:,4:], dtype=np.float32)
        # print(ac_state[0])
        combined_ac_state = np.hstack((
            (ac_state[:,[3, 5, 0, 1, 2, 6]] - np.array([SPD_BIAS, 0, 0, 0, ALT_BIAS, 0]))
            / np.array([SPD_SCALE_DOWN, TRACK_RATE_SCALE_DOWN, X_Y_SCALE_DOWN * PX_PER_NM, X_Y_SCALE_DOWN * PX_PER_NM, ALT_SCALE_DOWN, ALT_RATE_SCALE_DOWN]),
            np.sin(np.radians(ac_state[:,[4]])), np.cos(np.radians(ac_state[:,[4]])),
            np.sin(np.radians(ac_state[:,[8]])), np.cos(np.radians(ac_state[:,[8]])),
            (ac_state[:,[7, 9, 10]] - np.array([0, SPD_BIAS, APP_SPD_BIAS])) / np.array([ALT_SCALE_DOWN, SPD_SCALE_DOWN, APP_SPD_SCALE_DOWN]),
            ac_type_one_hot,
            ac_state[:,[14, 12, 15]]
        ))
        # print(combined_ac_state.shape)
        return combined_ac_state

    @staticmethod
    def _get_rewards_from_aircraft_state(aircraft_state) -> np.ndarray:
        # Also include agent ID for PettingZoo env
        return np.array(aircraft_state).reshape(AIRCRAFT_COUNT, -1)[:,[0, 17, 20]].astype(np.float32)

    @staticmethod
    def _get_terminated_from_aircraft_state(aircraft_state) -> np.ndarray:
        # Also include agent ID for PettingZoo env
        return np.array(aircraft_state).reshape(AIRCRAFT_COUNT, -1)[:,[18, 17, 20]].astype(np.int32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if not self.signalled_ready:
            self.sim_bridge.signal_trainer_initialized()
            self.signalled_ready = True

        # Send reset signal to simulator
        # print(f"[{self.instance_name}] Resetting")
        self.sim_bridge.signal_reset_sim()
        # Wait for simulator to signal ready for next action
        if self.episode % self.reset_print_period == 0:
            if self.episode > 0:
                print(f"[{self.instance_name}] {self.terminated_count} / {self.reset_print_period} episodes terminated before max_steps")
            print(f"[{self.instance_name}] Waiting for action ready after reset: episode {self.episode}")
            self.terminated_count = 0

            # Print distribution stats
            if self.action_dist:
                counts = pd.DataFrame(self.action_dist).apply(lambda x: x.value_counts(), axis=0).fillna(0).to_numpy()
                action_top_k = torch.tensor(counts).topk(min(5, len(counts)), dim=0)
                # print(action_top_k.indices)
                # print(action_top_k.values / len(self.action_dist))
            self.action_dist.clear()
        # print(f"[{self.instance_name}] Waiting reset ready")
        self.sim_bridge.wait_action_ready()
        # print(f"[{self.instance_name}] Reset action ready")

        # Get state from shared memory
        values = self.sim_bridge.get_aircraft_state()
        obs = self._get_observation_from_aircraft_state(values)

        info = {
            'step_offset': 0,
            'landing_rate': 0,
            'aircraft_conflict_rate_no_loc': 0,
            'mva_conflict_rate': 0,
            'wake_conflict_rate_no_loc': 0,
            'aircraft_conflict_rate_loc': 0,
            'wake_conflict_rate_loc': 0,
            'aircraft_conflict_rate': 0,
            'wake_conflict_rate': 0,
        }

        self.episode += 1
        self.steps = 0
        self.raw_steps = 0
        return obs, info

    def step(self, action):
        # Expects action of shape (AIRCRAFT_COUNT, 4)
        # (Heading, altitude, speed, action mask) for each aircraft

        # Validate that simulator is ready to accept action (proceed flag)
        values = self.sim_bridge.get_total_state()
        proceed_flag = values[0]
        if proceed_flag != 1:
            raise ValueError(f"[{self.instance_name}] Proceed flag must be 1")

        # Write action to shared memory and signal
        self.sim_bridge.write_actions(action)

        # Set the reset request flag before signaling action done
        # The next time the game loop finishes simulating max_steps frames, it will stop the update till reset() is called here
        # Handle step count offset (for conflict avoidance algorithm)
        self.steps += 1 + values[2]
        self.raw_steps += 1
        truncated = ((self.max_steps is not None and self.steps >= self.max_steps)
                     or (self.max_raw_steps is not None and self.raw_steps >= self.max_raw_steps))
        if truncated:
            # print(f"Truncating {self.instance_name} - steps: {self.steps}, raw_steps: {self.raw_steps}")
            self.sim_bridge.signal_reset_after_step()

        # print(int(time.time() * 1000), "Signalled action done")
        self.sim_bridge.signal_action_done()

        # print(f"{time.time()} Waiting for action ready")

        # Wait till simulator finished simulating 300 frames (action_ready event)
        # print("Waiting for simulation complete")
        self.sim_bridge.wait_action_ready()

        # Read state, reward, terminated, truncated from shared memory
        values = self.sim_bridge.get_total_state()
        aircraft_state = values[9 + AIRCRAFT_COUNT * (len(self.action_space.nvec) + 1):]
        obs = self._get_observation_from_aircraft_state(aircraft_state)
        reward = self._get_rewards_from_aircraft_state(aircraft_state)
        terminated = self._get_terminated_from_aircraft_state(aircraft_state)
        if terminated[obs[:,-2].astype(np.bool)][:,0].all():
            self.terminated_count += 1

        info = {
            'step_offset': values[2],
            'landing_rate': values[3],
            'aircraft_conflict_rate_no_loc': values[4],
            'mva_conflict_rate': values[5],
            'wake_conflict_rate_no_loc': values[6],
            'aircraft_conflict_rate_loc': values[7],
            'wake_conflict_rate_loc': values[8],
            'aircraft_conflict_rate': values[4] + values[7],
            'wake_conflict_rate': values[6] + values[8],
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        self.sim_bridge.close()
        if self.init_sim:
            print(f"[{self.instance_name}] Ending simulator process")
            self.sim_process.send_signal(signal.CTRL_C_EVENT if platform.system() == "Windows" else signal.SIGINT)
