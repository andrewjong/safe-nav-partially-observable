import warnings
from itertools import product
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C

from simulator.static_smoke import StaticSmoke, SmokeBlobParams

from learning.base_model import BaseModel

class Kernel:
    RBF = RBF(length_scale=0.45)
    Matern = Matern(length_scale=5.0)
    ConstantKernel = C()

N_RESTARTS_OPTIMIZER = 10
NORMALIZE_Y = False

class GaussianProcess(BaseModel):
    def __init__(self, kernel: Kernel = Kernel.Matern):
        super().__init__()

        self.kernel = kernel
        self.model= GaussianProcessRegressor(kernel=self.kernel, 
                                             optimizer='fmin_l_bfgs_b', 
                                             n_restarts_optimizer=N_RESTARTS_OPTIMIZER, 
                                             normalize_y=NORMALIZE_Y)
        
    def update(self):
        if self.input_history and self.output_history:
            X = np.array(self.input_history)
            y = np.array(self.output_history)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model.fit(X, y)

    def predict(self, x):
        y_pred, std = self.model.predict(x, return_std=True)
        return y_pred, std
    

    def score(self, x, y_true):
        y_pred, _ = self.predict(x)
        return mean_squared_error(y_true, y_pred)
    
    def plot_map(self, x_size, y_size, fig: plt.Figure = None, ax: plt.Axes = None):
        x = np.linspace(0, x_size, x_size)
        y = np.linspace(0, y_size, y_size)
        xy = np.array(list(product(y, x)))

        y_pred, std = self.predict(xy)
        y_pred = y_pred.reshape(y_size, x_size)

        if fig is None or ax is None:
            fig, ax = plt.subplots()

        if ax.images:
            ax.images[0].set_array(y_pred)
        else:
            ax_ = ax.imshow(y_pred, cmap='gray', vmin=0, vmax=1.0, extent=[0, x_size, 0, y_size], origin='lower')
            ax.set_title('Predicted Map')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            #fig.colorbar(ax_, label='Estimated Smoke Density', shrink=0.5)

        fig.canvas.draw()

    
if __name__ == '__main__':
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

    x = np.linspace(0, x_size, x_size)
    y = np.linspace(0, y_size, y_size)
    xy = np.array(list(product(y, x)))

    y_pred, std = gp.predict(xy)
    y_pred = y_pred.reshape(y_size, x_size)
    std = std.reshape(y_size, x_size)
    y_true = smoke_simulator.get_smoke_map()

    f, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(y_true, vmin=0, vmax=y_true.max(), extent=[0, x_size, 0, y_size], origin='lower', cmap='gray')
    ax[0].set_title('Ground truth')
    ax[1].imshow(y_pred, vmin=0, vmax=y_pred.max(), origin='lower', cmap='gray')
    ax[1].set_title('Predicted')
    ax[2].imshow(std, vmin=0, vmax=std.max(), origin='lower', cmap='gray')
    ax[2].set_title('Std')

    plt.show()

    fig, ax = plt.subplots()
    gp.plot_map(x_size, y_size, fig=fig, ax=ax)
    plt.show()

