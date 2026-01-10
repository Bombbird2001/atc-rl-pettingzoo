import gymnasium as gym
import numpy as np
import os
import pandas as pd
import platform
import random
import signal
import subprocess
import torch
from common.constants import AIRCRAFT_COUNT, SPD_BIAS, SPD_SCALE_DOWN, \
    TRACK_RATE_SCALE_DOWN, X_Y_SCALE_DOWN, PX_PER_NM, ALT_SCALE_DOWN, \
    ALT_RATE_SCALE_DOWN, ALT_BIAS
from common.data_preprocessing import RECAT_MAPPING
from gymnasium import spaces
from sklearn.preprocessing import OneHotEncoder
from utils.game_bridge import GameBridge


SIMULATOR_JAR = os.getenv("SIMULATOR_JAR")


class TC2GymEnv(gym.Env):
    def __init__(
            self, ac_type_one_hot_encoder: OneHotEncoder, is_eval=False, render_mode=None, reset_print_period=1, instance_suffix="",
            init_sim=True, max_steps=300
    ):
        super().__init__()

        if init_sim:
            instance_suffix = f"{instance_suffix}_{random.randbytes(3).hex()}"
        self.init_sim = init_sim
        self.sim_bridge = GameBridge.get_bridge_for_platform(instance_suffix=instance_suffix)
        self.signalled_ready = False

        self.instance_name = f"env{instance_suffix}"

        self.is_eval = is_eval
        self.reset_print_period = reset_print_period

        # Actions[0] = [aircraft 0 to 14, or no clearance (value = 0)]
        # Actions[1] = [steps of 5 degrees from 0-359]
        # Actions[2] = [steps of 1000 feet from min to max altitude - 2000 to FL150 for Singapore]
        # Actions[3] = [steps of 10 knots from 160 to 250 knots (for now)]
        self.action_space = spaces.MultiDiscrete([72, 14, 10])

        # [aircraft type, x, y, alt, gs, track, angular speed, vertical speed,
        # current cleared altitude, current cleared heading, current cleared speed, localizer captured] normalized
        # +1 for aircraft masking
        self.OBS_SPACE_DIMENSION = 19
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
        self.terminated_count = 0
        self.render_mode = render_mode

        self.action_dist = []

        print(f"[{self.instance_name}] Environment initialized")

        if init_sim:
            print(f"[{self.instance_name}] Starting simulator")
            self.sim_process = subprocess.Popen(f"java -jar \"{SIMULATOR_JAR}\" {instance_suffix}", shell=True)

    def _get_observation_from_aircraft_state(self, aircraft_state) -> np.ndarray:
        tmp_state = np.array(aircraft_state).reshape(AIRCRAFT_COUNT, -1)[:,1:]
        # print(tmp_state)
        ac_types = np.array(tmp_state[:,:4], dtype=np.str_)
        ac_types = [[RECAT_MAPPING.get(ac_type, "Unknown")] for ac_type in (ac_types[:,0] + ac_types[:,1] + ac_types[:,2] + ac_types[:,3])]
        ac_type_one_hot = self.ac_type_one_hot_encoder.transform(ac_types).toarray()
        # Map
        # ICAO type, x, y, alt, ias, track, track rate, vertical speed, cleared alt, cleared hdg, cleared IAS, LOC cap, mask, terminated, agent ID
        # to
        # ["ias", "track_rate", "x", "y", "combined_alt", "combined_alt_rate", "track_x", "track_y", "prev_cleared_hdg_x", "prev_cleared_hdg_y",
        # "prev_cleared_alt", "prev_cleared_ias"] + [f"aircraft_type_{j}" for j in range(aircraft_category_count)] + ["mask", "agent ID"]
        ac_state = np.array(tmp_state[:,4:], dtype=np.float32)
        # print(ac_state[0])
        combined_ac_state = np.hstack((
            (ac_state[:,[3, 5, 0, 1, 2, 6]] - np.array([SPD_BIAS, 0, 0, 0, ALT_BIAS, 0]))
            / np.array([SPD_SCALE_DOWN, TRACK_RATE_SCALE_DOWN, X_Y_SCALE_DOWN * PX_PER_NM, X_Y_SCALE_DOWN * PX_PER_NM, ALT_SCALE_DOWN, ALT_RATE_SCALE_DOWN]),
            np.sin(np.radians(ac_state[:,[4]])), np.cos(np.radians(ac_state[:,[4]])),
            np.sin(np.radians(ac_state[:,[8]])), np.cos(np.radians(ac_state[:,[8]])),
            (ac_state[:,[7, 9]] - np.array([0, SPD_BIAS])) / np.array([ALT_SCALE_DOWN, SPD_SCALE_DOWN]),
            ac_type_one_hot,
            ac_state[:,[11, 13]]
        ))
        # print(combined_ac_state.shape)
        return combined_ac_state

    @staticmethod
    def _get_rewards_from_aircraft_state(aircraft_state) -> np.ndarray:
        # Also include agent ID for PettingZoo env
        return np.array(aircraft_state).reshape(AIRCRAFT_COUNT, -1)[:,[0, 16, 18]].astype(np.float32)

    @staticmethod
    def _get_terminated_from_aircraft_state(aircraft_state) -> np.ndarray:
        # Also include agent ID for PettingZoo env
        return np.array(aircraft_state).reshape(AIRCRAFT_COUNT, -1)[:,[17, 16, 18]].astype(np.int32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if not self.signalled_ready:
            self.sim_bridge.signal_trainer_initialized()
            self.signalled_ready = True

        # Send reset signal to simulator
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
        self.sim_bridge.wait_action_ready()

        # Get state from shared memory
        values = self.sim_bridge.get_aircraft_state()
        obs = self._get_observation_from_aircraft_state(values)

        info = {}
        self.episode += 1
        self.steps = 0
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
        # self.action_dist.append(action) TODO update for MARL

        self.sim_bridge.write_actions(action)

        # Set the reset request flag before signalling action done
        # The next time the game loop finishes simulating 300 frames, it will stop the update till reset() is called here
        self.steps += 1
        truncated = self.max_steps is not None and self.steps >= self.max_steps
        if truncated and not self.is_eval:
            # print(f"Truncating={truncated}")
            self.sim_bridge.signal_reset_after_step()

        # print(int(time.time() * 1000), "Signalled action done")
        self.sim_bridge.signal_action_done()

        # print(f"{time.time()} Waiting for action ready")

        # Wait till simulator finished simulating 300 frames (action_ready event)
        # print("Waiting for simulation complete")
        self.sim_bridge.wait_action_ready()

        # Read state, reward, terminated, truncated from shared memory
        values = self.sim_bridge.get_total_state()
        aircraft_state = values[1 + AIRCRAFT_COUNT * (len(self.action_space.nvec) + 1):]
        obs = self._get_observation_from_aircraft_state(aircraft_state)
        reward = self._get_rewards_from_aircraft_state(aircraft_state)
        terminated = self._get_terminated_from_aircraft_state(aircraft_state)
        if terminated[:,0].all():
            self.terminated_count += 1

        info = {}

        return obs, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        self.sim_bridge.close()
        if self.init_sim:
            print(f"[{self.instance_name}] Ending simulator process")
            self.sim_process.send_signal(signal.CTRL_C_EVENT if platform.system() == "Windows" else signal.SIGINT)
