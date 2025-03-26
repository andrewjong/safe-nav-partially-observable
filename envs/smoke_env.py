import numpy as np
import gymnasium as gym
from gymnasium import spaces
from simulator.static_smoke import StaticSmoke, SmokeBlobParams
from agents.basic_robot import BasicRobot, RobotParams
from dataclasses import dataclass, field
import matplotlib.pyplot as plt

@dataclass
class EnvParams:
    world_x_size: int = field(default=50)
    world_y_size: int = field(default=50)

    max_steps: int = field(default=1000)
    render: bool = field(default=False)
    clock: float = field(default=0.1)

class SmokeEnv(gym.Env):
    def __init__(self, env_params: EnvParams, robot_params: RobotParams, smoke_blob_params: list[SmokeBlobParams]) -> None:
        super().__init__()

        self.env_params = env_params
        self.robot_params = robot_params
        self.smoke_blob_params = smoke_blob_params

        self.smoke_simulator = StaticSmoke(env_params.world_x_size, env_params.world_y_size, smoke_blob_params)

        self.robot_params.world_x_size = self.env_params.world_x_size
        self.robot_params.world_y_size = self.env_params.world_y_size

        for blob in self.smoke_blob_params:
            blob.world_x_size = self.env_params.world_x_size
            blob.world_y_size = self.env_params.world_y_size


        self.action_space = spaces.Box(low=np.array([self.robot_params.v_min, self.robot_params.omega_min]),
                                       high=np.array([self.robot_params.v_max, self.robot_params.omega_max]),
                                       shape=(2,))
        
        # 0: x, 1: y, 2: theta, 3: smoke_density
        self.observation_space = spaces.Box(low=np.array([0, 0, 0, 0]),
                                             high=np.array([self.env_params.world_x_size, self.env_params.world_y_size, 2*np.pi, 1.]),
                                             shape=(4,))

        self.robot = BasicRobot(robot_params)
        self.window = {"fig": None, "ax": None, "cax": None}
        self.clock = self.env_params.clock if self.env_params.render else None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.window = {"fig": None, "ax": None, "cax": None}
        obs = self.observation_space.sample()

        self.robot.reset(obs[0], obs[1], obs[2])

        # TODO: reset smoke simulator
        #self.smoke_simulator.reset()

        return self._get_obs(), {}
        
    def _get_obs(self):
        return np.array([self.robot.pos_x, self.robot.pos_y, self.robot.angle, self.smoke_simulator.get_smoke_density(self.robot.pos_x, self.robot.pos_y)])

    def _get_info(self):
        return {}

    def _get_reward(self, obs, action):
        return 0.

    def _get_terminated(self, obs):
        return False
    
    def _get_truncated(self, obs):
        return False

    def step(self, action):
        self.robot.dynamic_step(action[0], action[1])

        # TODO: update smoke simulator
        #self.smoke_simulator.step()

        obs = self._get_obs()
        reward = self._get_reward(obs, action)
        terminated = self._get_terminated(obs)
        truncated = self._get_truncated(obs)
        info = self._get_info()

        if self.env_params.render:
            self._render_frame()

        return obs, reward, terminated, truncated, info
    
    def _render_frame(self):
        if self.window["fig"] is None:
            self.window["fig"], self.window["ax"] = plt.subplots()
            self.window["cax"] = self.window["ax"].imshow(
                self.smoke_simulator.get_smoke_map(),
                cmap='gray'
            )
            self.window["ax"].set_axis_off()
            self.window["fig"].colorbar(self.window["cax"], ax=self.window["ax"], label="Smoke Density")
            self.window["cax"].set_clim(vmin=np.min(0.0),
                                        vmax=np.max(1.0))
            
        for arrow in self.window["ax"].patches:
            arrow.remove()

        # Plot the agent's location as a blue arrow
        self.window["ax"].arrow(self.robot.pos_x, self.robot.pos_y, np.cos(self.robot.angle), np.sin(self.robot.angle), 
                                head_width=0.5, head_length=0.5, fc='b', ec='b')
        
        # Redraw the plot to update the frame
        plt.draw()
        plt.pause(self.clock)

    def close(self):
        self.window = {"fig": None, "ax": None, "cax": None}

if __name__ == "__main__":
    env_params = EnvParams()
    env_params.world_x_size = 50
    env_params.world_y_size = 50
    env_params.max_steps = 100
    env_params.render = True

    robot_params = RobotParams()
    smoke_blob_params = [
        SmokeBlobParams(x_pos=10, y_pos=40, intensity=2.0, spread_rate=1.0),
        SmokeBlobParams(x_pos=20, y_pos=20, intensity=1.5, spread_rate=3.0),
        SmokeBlobParams(x_pos=15, y_pos=45, intensity=2.0, spread_rate=4.0),
        SmokeBlobParams(x_pos=40, y_pos=40, intensity=2.5, spread_rate=4.0)
    ]

    env = SmokeEnv(env_params, robot_params, smoke_blob_params)
    env.reset()

    for _ in range(100):
        state, reward, terminated, truncated, info = env.step(env.action_space.sample())
        print(state[3])

    env.close()
