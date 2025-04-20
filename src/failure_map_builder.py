import numpy as np
from dataclasses import dataclass
from learning.base_model import BaseModel
from learning.gaussian_process import GaussianProcess
from itertools import product
import matplotlib.pyplot as plt
from simulator.static_smoke import StaticSmoke, SmokeBlobParams

@dataclass
class FailureMapParams:
    x_size: int
    y_size: int
    resolution: float

    # Available map rule types:
    # - 'threshold': threshold map rule
    # TODO: Add more map rule types
    map_rule_type: str

    # Used for threshold map rule
    map_rule_threshold: float


class GPFailureMapBuilder():  
    def __init__(self, params: FailureMapParams):
        """
        Builds a failure map based on the given parameters.

        Args:
            params: FailureMapParams
        """
        self.params = params

        # TODO: Check if the initial map is correct?
        self.failure_map = np.ones((int(self.params.y_size / self.params.resolution), int(self.params.x_size / self.params.resolution)))

        x = np.linspace(0, self.params.x_size, int(self.params.x_size / self.params.resolution))
        y = np.linspace(0, self.params.y_size, int(self.params.y_size / self.params.resolution))
        self.xy_coords = np.array(list(product(y, x)))

    def build_map(self, forecaster: BaseModel):
        """
        Builds a failure map based on the given forecaster. Returns the failure map in the form of a numpy array filled with 0s and 1s.

        Args:
            forecaster: BaseModel
        Returns:
            failure_map: np.ndarray
        """
        try:
            y_pred, _ = forecaster.predict(self.xy_coords)
            continuous_map = y_pred.reshape(int(self.params.y_size / self.params.resolution), int(self.params.x_size / self.params.resolution))

            self.failure_map = self.rule_based_map(continuous_map)
        except Exception as e:
            print(f"Error predicting: {e}. Returning last failure map.")

        return self.failure_map
        
    def build_continuous_map(self, forecaster: BaseModel):
        try:
            y_pred, _ = forecaster.predict(self.xy_coords)
            continuous_map = y_pred.reshape(int(self.params.y_size / self.params.resolution), int(self.params.x_size / self.params.resolution))
            return continuous_map
        except Exception as e:
            print(f"Error predicting: {e}. Returning last failure map instead.")
            return self.failure_map
    
    def rule_based_map(self, continuous_map: np.ndarray):
        if self.params.map_rule_type == 'threshold':
            failure_map = (continuous_map < self.params.map_rule_threshold).astype(bool)
        else:
            raise ValueError(f"Invalid map rule type: {self.params.map_rule_type}")

        return failure_map
    
    def plot_failure_map(self, fig: plt.Figure = None, ax: plt.Axes = None):
        if fig is None or ax is None:
            fig, ax = plt.subplots()

        if ax.images:
            ax.images[0].set_array(self.failure_map)
        else:
            ax_ = ax.imshow(self.failure_map, vmin=0.0, vmax=1.0, extent=[0, self.params.x_size, 0, self.params.y_size], origin='lower', cmap='gray')
            #fig.colorbar(ax_, label='Failure', shrink=0.5)
            ax.set_title('Failure Map')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
        
        fig.canvas.draw()


if __name__ == "__main__":
    params = FailureMapParams(x_size=80, y_size=50, resolution=1, map_rule_type='threshold', map_rule_threshold=0.7)
    builder = GPFailureMapBuilder(params)
    print("Initial failure map shape:", builder.failure_map.shape)
    builder.plot_failure_map()

    x_size, y_size = 80, 50

    smoke_blob_params = [
        SmokeBlobParams(x_pos=10, y_pos=40, intensity=2.0, spread_rate=4.0),
        SmokeBlobParams(x_pos=10, y_pos=20, intensity=1.5, spread_rate=5.0),
        SmokeBlobParams(x_pos=60, y_pos=20, intensity=2.0, spread_rate=3.0),
        SmokeBlobParams(x_pos=50, y_pos=10, intensity=1.5, spread_rate=5.0),
    ]

    smoke_simulator = StaticSmoke(x_size=x_size, y_size=y_size, resolution=0.1, smoke_blob_params=smoke_blob_params)

    gp = GaussianProcess()

    sample_size = 100

    for i in range(sample_size):
        X_sample = np.concatenate([np.random.uniform(0, y_size, 1), np.random.uniform(0, x_size, 1)])
        y_observe = smoke_simulator.get_smoke_density(X_sample[1], X_sample[0])
        gp.track_data(X_sample, y_observe)
    gp.update()

    y_true = smoke_simulator.get_smoke_map()

    plt.figure()
    plt.imshow(y_true, vmin=0.0, vmax=1.0, extent=[0, x_size, 0, y_size], origin='lower', cmap='gray')
    plt.show()

    failure_map = builder.build_map(gp)
    print("Failure map shape:", failure_map.shape)
    fig, ax = plt.subplots()
    builder.plot_failure_map(fig=fig, ax=ax)
    plt.show()