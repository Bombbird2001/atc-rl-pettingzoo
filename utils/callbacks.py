from stable_baselines3.common.callbacks import BaseCallback
from typing import Callable, Dict


class PPOStatsCallback(BaseCallback):
    def __init__(self, log_stats: Callable[[Dict], None], log_interval=1000, verbose=1):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.log_stats = log_stats

    def _on_step(self) -> bool:
        if len(self.locals["infos"]) > 0:
            for info in self.locals["infos"]:
                if "episode" in info.keys():
                    self.log_stats({
                        "episode/reward": info["episode"]["r"],
                        "episode/length": info["episode"]["l"],
                        "episode/cumulative_time": info["episode"]["t"],
                    })

        return True
