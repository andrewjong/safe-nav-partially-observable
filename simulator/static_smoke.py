import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt

@dataclass
class SmokeBlobParams:
    x_pos: int
    y_pos: int
    intensity: float
    spread_rate: float

class StaticSmoke:
    def __init__(self, x_size, y_size, smoke_blob_params: list[SmokeBlobParams]):
        self.x_size = x_size
        self.y_size = y_size

        self.smoke_map = self.build_smoke_map(smoke_blob_params)

    def build_smoke_map(self, smoke_blob_params: list[SmokeBlobParams]):
        smoke_map = np.zeros((self.x_size, self.y_size))

        for blob in smoke_blob_params:
            y, x = np.ogrid[-blob.y_pos:smoke_map.shape[0]-blob.y_pos,
                            -blob.x_pos:smoke_map.shape[1]-blob.x_pos]
            
            sigma = blob.spread_rate
            gaussian = blob.intensity * np.exp(-(x*x + y*y)/(2.0*sigma**2))
            
            smoke_map += gaussian

        return smoke_map
        
    def get_smoke_density(self, x, y):
        x = int(np.clip(x, 0, self.x_size - 1))
        y = int(np.clip(y, 0, self.y_size - 1))
        return self.smoke_map[x, y]

    def get_smoke_map(self):
        return self.smoke_map
    
    def plot_smoke_map(self):
        plt.imshow(self.smoke_map, cmap='gray')
        plt.colorbar(label='Smoke Density')
        plt.title('Static Smoke Map')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.show()


if __name__ == "__main__":
    smoke_blob_params = [
        SmokeBlobParams(x_pos=10, y_pos=40, intensity=2.0, spread_rate=1.0),
        SmokeBlobParams(x_pos=20, y_pos=20, intensity=1.5, spread_rate=3.0),
        SmokeBlobParams(x_pos=15, y_pos=45, intensity=2.0, spread_rate=4.0),
        SmokeBlobParams(x_pos=40, y_pos=40, intensity=2.5, spread_rate=4.0)
    ]
    
    smoke_simulator = StaticSmoke(x_size=50, y_size=50, smoke_blob_params=smoke_blob_params)
    smoke_simulator.plot_smoke_map()






