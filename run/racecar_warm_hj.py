import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import math

import posggym
from reachability.warm_start_solver import (WarmStartSolver,
                                            WarmStartSolverConfig)
from src.mppi import Navigator, dubins_dynamics_tensor


class OccupancyMap:
    """
    A class to build and maintain an occupancy grid map from lidar observations.
    
    The map is represented as a 2D grid where each cell can be in one of three states:
    - UNSEEN: Cell has not been observed yet
    - FREE: Cell has been observed and is free (no obstacle)
    - OCCUPIED: Cell has been observed and contains an obstacle
    
    Attributes:
        width (int): Width of the map in world units
        height (int): Height of the map in world units
        resolution (float): Size of each grid cell in world units
        grid_width (int): Width of the grid in cells
        grid_height (int): Height of the grid in cells
        grid (numpy.ndarray): The occupancy grid
        UNSEEN (int): Value representing an unseen cell
        FREE (int): Value representing a free cell
        OCCUPIED (int): Value representing an occupied cell
    """
    
    # Cell state constants
    UNSEEN = 0
    FREE = 1
    OCCUPIED = 2
    
    def __init__(self, width, height, resolution):
        """
        Initialize the occupancy map.
        
        Args:
            width (int): Width of the map in world units
            height (int): Height of the map in world units
            resolution (float): Size of each grid cell in world units
        """
        self.width = width
        self.height = height
        self.resolution = resolution
        
        # Calculate grid dimensions
        self.grid_width = int(np.ceil(width / resolution))
        self.grid_height = int(np.ceil(height / resolution))
        
        # Initialize grid with all cells marked as UNSEEN
        self.grid = np.zeros((self.grid_height, self.grid_width), dtype=np.uint8)
        
        # For visualization
        self.fig = None
        self.ax = None
        self.map_img = None
        self.robot_marker = None
        self.lidar_lines = []
        
        # Last known robot position for visualization
        self.last_robot_pos = None
        self.last_robot_angle = None
    
    def world_to_grid(self, x, y):
        """
        Convert world coordinates to grid indices.
        
        Args:
            x (float): X coordinate in world units
            y (float): Y coordinate in world units
            
        Returns:
            tuple: (grid_row, grid_col) indices
        """
        grid_col = int(x / self.resolution)
        grid_row = int(y / self.resolution)
        
        # Ensure indices are within grid bounds
        grid_row = max(0, min(grid_row, self.grid_height - 1))
        grid_col = max(0, min(grid_col, self.grid_width - 1))
        
        return grid_row, grid_col
    
    def grid_to_world(self, row, col):
        """
        Convert grid indices to world coordinates (center of cell).
        
        Args:
            row (int): Grid row index
            col (int): Grid column index
            
        Returns:
            tuple: (x, y) coordinates in world units
        """
        x = (col + 0.5) * self.resolution
        y = (row + 0.5) * self.resolution
        return x, y
    
    def is_in_bounds(self, row, col):
        """
        Check if grid indices are within the grid bounds.
        
        Args:
            row (int): Grid row index
            col (int): Grid column index
            
        Returns:
            bool: True if indices are within bounds, False otherwise
        """
        return 0 <= row < self.grid_height and 0 <= col < self.grid_width
    
    def update_from_lidar(self, lidar_distances, vehicle_x, vehicle_y, vehicle_angle):
        """
        Update the occupancy map based on lidar observations.
        
        Args:
            lidar_distances (numpy.ndarray): Array of lidar distance readings
            vehicle_x (float): X coordinate of the vehicle in world units
            vehicle_y (float): Y coordinate of the vehicle in world units
            vehicle_angle (float): Orientation of the vehicle in radians
        """
        # Store robot position for visualization
        self.last_robot_pos = (vehicle_x, vehicle_y)
        self.last_robot_angle = vehicle_angle
        
        # Get robot position in grid coordinates
        robot_row, robot_col = self.world_to_grid(vehicle_x, vehicle_y)
        
        # Number of lidar beams
        n_sensors = len(lidar_distances)
        
        # Angle between consecutive lidar beams
        angle_inc = 2 * math.pi / n_sensors
        
        # Maximum distance to trace along each beam
        max_dist = 5.0  # Assuming this is the max lidar range
        
        # Process each lidar beam
        for i, distance in enumerate(lidar_distances):
            # Calculate beam angle in world frame
            beam_angle = vehicle_angle + i * angle_inc
            
            # Normalize angle to [0, 2*pi)
            beam_angle = beam_angle % (2 * math.pi)
            
            # Calculate end point of beam
            if distance >= max_dist:
                # No obstacle detected, beam reaches max range
                end_x = vehicle_x + max_dist * math.cos(beam_angle)
                end_y = vehicle_y + max_dist * math.sin(beam_angle)
                obstacle_detected = False
            else:
                # Obstacle detected at distance
                end_x = vehicle_x + distance * math.cos(beam_angle)
                end_y = vehicle_y + distance * math.sin(beam_angle)
                obstacle_detected = True
            
            # Convert end point to grid coordinates
            end_row, end_col = self.world_to_grid(end_x, end_y)
            
            # Use Bresenham's line algorithm to trace the beam
            cells = self.bresenham_line(robot_row, robot_col, end_row, end_col)
            
            # Mark cells along the beam as free, except the last one if obstacle detected
            for j, (row, col) in enumerate(cells):
                if j == len(cells) - 1 and obstacle_detected:
                    # Last cell contains obstacle
                    self.grid[row, col] = self.OCCUPIED
                else:
                    # Intermediate cells are free
                    self.grid[row, col] = self.FREE
    
    def bresenham_line(self, row1, col1, row2, col2):
        """
        Implementation of Bresenham's line algorithm to get all grid cells along a line.
        
        Args:
            row1, col1 (int): Starting grid cell indices
            row2, col2 (int): Ending grid cell indices
            
        Returns:
            list: List of (row, col) tuples representing grid cells along the line
        """
        cells = []
        
        # Calculate differences and step directions
        d_row = abs(row2 - row1)
        d_col = abs(col2 - col1)
        
        # Determine step direction
        s_row = 1 if row1 < row2 else -1
        s_col = 1 if col1 < col2 else -1
        
        # Error value
        err = d_col - d_row
        
        # Current position
        row, col = row1, col1
        
        # Iterate until we reach the end point
        while True:
            # Add current cell if it's within bounds
            if self.is_in_bounds(row, col):
                cells.append((row, col))
            
            # Check if we've reached the end
            if row == row2 and col == col2:
                break
            
            # Calculate error for next step
            e2 = 2 * err
            
            # Update position and error
            if e2 > -d_row:
                err -= d_row
                col += s_col
            
            if e2 < d_col:
                err += d_col
                row += s_row
        
        return cells
    
    def initialize_plot(self):
        """Initialize the matplotlib figure for visualization."""
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            plt.ion()  # Enable interactive mode
            
            # Create initial image
            cmap = plt.cm.colors.ListedColormap(['gray', 'white', 'black'])
            bounds = [-0.5, 0.5, 1.5, 2.5]
            norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
            
            self.map_img = self.ax.imshow(self.grid, cmap=cmap, norm=norm, origin='lower')
            
            # Add colorbar
            cbar = self.fig.colorbar(self.map_img, ticks=[0, 1, 2])
            cbar.ax.set_yticklabels(['Unseen', 'Free', 'Occupied'])
            
            # Set axis labels
            self.ax.set_xlabel('X (grid cells)')
            self.ax.set_ylabel('Y (grid cells)')
            self.ax.set_title('Occupancy Map')
            
            # Add robot marker
            if self.last_robot_pos is not None:
                robot_row, robot_col = self.world_to_grid(*self.last_robot_pos)
                self.robot_marker = self.ax.plot([robot_col], [robot_row], 'ro', markersize=10)[0]
            
            self.fig.canvas.draw()
            plt.pause(0.001)
    
    def update_plot(self):
        """Update the matplotlib visualization with current map data."""
        if self.fig is None:
            self.initialize_plot()
        else:
            # Update map data
            self.map_img.set_data(self.grid)
            
            # Update robot position
            if self.last_robot_pos is not None:
                robot_row, robot_col = self.world_to_grid(*self.last_robot_pos)
                if self.robot_marker is None:
                    self.robot_marker = self.ax.plot([robot_col], [robot_row], 'ro', markersize=10)[0]
                else:
                    self.robot_marker.set_data([robot_col], [robot_row])
                
                # Clear previous lidar lines
                for line in self.lidar_lines:
                    line.remove()
                self.lidar_lines = []
                
                # Draw lidar lines for visualization
                if self.last_robot_angle is not None:
                    n_sensors = 16  # Assuming 16 sensors as in the main code
                    angle_inc = 2 * math.pi / n_sensors
                    max_dist = 5.0
                    
                    for i in range(n_sensors):
                        beam_angle = self.last_robot_angle + i * angle_inc
                        end_x = self.last_robot_pos[0] + max_dist * math.cos(beam_angle)
                        end_y = self.last_robot_pos[1] + max_dist * math.sin(beam_angle)
                        
                        end_row, end_col = self.world_to_grid(end_x, end_y)
                        line = self.ax.plot([robot_col, end_col], [robot_row, end_row], 'r-', alpha=0.3)[0]
                        self.lidar_lines.append(line)
            
            self.fig.canvas.draw()
            plt.pause(0.001)
    
    def plot(self):
        """Plot the occupancy map."""
        self.update_plot()


def main():

    MAP_WIDTH = 30
    MAP_HEIGHT = 30
    MAP_RESOLUTION = 1.0

    N_SENSORS = 16
    env = posggym.make('DrivingContinuous-v0', world="30x30ScatteredObstacleField", num_agents=1, n_sensors=N_SENSORS, render_mode="human")



    solver = WarmStartSolver(
        config=WarmStartSolverConfig(
            system_name="dubins3d",
            domain_cells=[MAP_WIDTH * MAP_RESOLUTION, MAP_HEIGHT * MAP_RESOLUTION, 40],
            domain=[[0, 0, 0], [MAP_WIDTH, MAP_HEIGHT, 2*np.pi]],
            mode="brt",
            accuracy="medium",
            converged_values=None,
            until_convergent=False,
            print_progress=False,
        )
    )

    # nom_controller = Navigator()
    # nom_controller.set_odom(state[:2],state[2])
    # nom_controller.set_map(builder.failure_map, [30, 40], [0, 0], 1.0)
    # nom_controller.set_goal(list(env_params.goal_location))

    occupancy_map = OccupancyMap(MAP_WIDTH, MAP_HEIGHT, MAP_RESOLUTION)

    observations, infos = env.reset()
    lidar_distances, vehicle_x, vehicle_y, vehicle_angle = observations["0"][0:N_SENSORS], observations["0"][2 * N_SENSORS], observations["0"][2 * N_SENSORS + 1], observations["0"][2*N_SENSORS + 2]

    occupancy_map.update_from_lidar(lidar_distances, vehicle_x, vehicle_y, vehicle_angle)
    occupancy_map.plot()  # Initialize and show the plot


    # creates a failure map with the given width, height and resolution.
    # unobserved cells are initialized as fail set.

    for _ in range(300):
        actions = {i: env.action_spaces[i].sample() for i in env.agents}
        observations, rewards, terminations, truncations, all_done, infos = env.step(actions)

        observation = observations["0"]

        lidar_distances = observation[0:N_SENSORS]
        vehicle_x = observation[2 * N_SENSORS]
        vehicle_y = observation[2 * N_SENSORS + 1]
        vehicle_angle = observation[2*N_SENSORS + 2]
        vehicle_x_velocity = observation[2*N_SENSORS + 3]
        vehicle_y_velocity = observation[2*N_SENSORS + 4]
        env.render()


        print(f"{vehicle_x=}, {vehicle_y=}, {vehicle_angle=}, {vehicle_x_velocity=}, {vehicle_y_velocity=}")

        # update the fail set from the lidar observations. cells that are free are marked as safe.
        # assumes the lidar observations are equally spaced from 0 to 2*pi
        occupancy_map.update_from_lidar(lidar_distances, vehicle_x, vehicle_y, vehicle_angle)
        occupancy_map.update_plot()  # Update the plot with new data


        if all_done:
            observations, infos = env.reset()

    env.close()
    plt.ioff()  # Turn off interactive mode when done
    plt.show()  # Show the final plot


if __name__ == "__main__":
    main()
