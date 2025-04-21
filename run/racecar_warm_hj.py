import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import math

import posggym
# Comment out reachability import since we're focusing on MPPI visualization
from reachability.warm_start_solver import (WarmStartSolver,
                                            WarmStartSolverConfig)
from src.mppi import Navigator, dubins_dynamics_tensor


MAP_WIDTH = 30
MAP_HEIGHT = 30
MAP_RESOLUTION = 1.0  # units per cell

N_SENSORS = 32
MAX_SENSOR_DISTANCE = 5.0
ROBOT_ORIGIN = [1,1]
robot_goal=[0,0]


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
        self.mppi_trajectory_lines = []
        self.chosen_trajectory_line = None
        
        # Last known robot position for visualization
        self.last_robot_pos = None
        self.last_robot_angle = None
        self.n_sensors = N_SENSORS
    
    def world_to_grid(self, x, y):
        """
        Convert world coordinates to grid indices.
        
        Args:
            x (float): X coordinate in world units
            y (float): Y coordinate in world units
            
        Returns:
            tuple: (grid_row, grid_col) indices
        """
        # Use proper rounding instead of truncation to get the nearest grid cell
        grid_col = int(round(x / self.resolution))
        grid_row = int(round(y / self.resolution))
        
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
        # Since we're using rounding in world_to_grid, we should return the center of the grid cell
        # without adding 0.5 (which was needed when using truncation)
        x = col * self.resolution
        y = row * self.resolution
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
    
    def update_from_lidar(self, lidar_distances, vehicle_x, vehicle_y, vehicle_angle, debug=False):
        """
        Update the occupancy map based on lidar observations.
        
        Args:
            lidar_distances (numpy.ndarray): Array of lidar distance readings
            vehicle_x (float): X coordinate of the vehicle in world units
            vehicle_y (float): Y coordinate of the vehicle in world units
            vehicle_angle (float): Orientation of the vehicle in radians
            debug (bool): Whether to print debug information
        """
        # Store robot position for visualization
        self.last_robot_pos = (vehicle_x, vehicle_y)
        self.last_robot_angle = vehicle_angle
        
        # Get robot position in grid coordinates
        robot_row, robot_col = self.world_to_grid(vehicle_x, vehicle_y)
        
        if debug:
            print(f"Robot world position: ({vehicle_x}, {vehicle_y})")
            print(f"Robot grid position: ({robot_row}, {robot_col})")

        # Number of lidar beams
        self.n_sensors = len(lidar_distances)
        
        # Angle between consecutive lidar beams
        angle_inc = 2 * math.pi / self.n_sensors
        
        # Process each lidar beam
        for i, distance in enumerate(lidar_distances):
            # Only process every 4th beam in debug mode to reduce output
            current_debug = debug and (i % 4 == 0)
            
            # Calculate beam angle in world frame
            beam_angle = vehicle_angle + i * angle_inc
            
            # Normalize angle to [0, 2*pi)
            beam_angle = beam_angle % (2 * math.pi)
            
            # Calculate end point of beam
            if distance >= MAX_SENSOR_DISTANCE:
                # No obstacle detected, beam reaches max range
                end_x = vehicle_x + MAX_SENSOR_DISTANCE * math.cos(beam_angle)
                end_y = vehicle_y + MAX_SENSOR_DISTANCE * math.sin(beam_angle)
                obstacle_detected = False
            else:
                # Obstacle detected at distance
                end_x = vehicle_x + distance * math.cos(beam_angle)
                end_y = vehicle_y + distance * math.sin(beam_angle)
                obstacle_detected = True
            
            # Convert end point to grid coordinates
            end_row, end_col = self.world_to_grid(end_x, end_y)
            
            if current_debug:
                print(f"Beam {i}: angle={beam_angle:.2f}, distance={distance:.2f}")
                print(f"  End point world: ({end_x:.2f}, {end_y:.2f})")
                print(f"  End point grid: ({end_row}, {end_col})")
                print(f"  Obstacle detected: {obstacle_detected}")
            
            # Use Bresenham's line algorithm to trace the beam
            cells = self.bresenham_line(robot_row, robot_col, end_row, end_col)
            
            if current_debug:
                print(f"  Cells along beam: {len(cells)}")
                if len(cells) > 0:
                    print(f"  First cell: {cells[0]}, Last cell: {cells[-1]}")
            
            # Mark cells along the beam as free, except the last one if obstacle detected
            for j, (row, col) in enumerate(cells):
                if j == len(cells) - 1 and obstacle_detected:
                    # Last cell contains obstacle
                    self.grid[row, col] = self.OCCUPIED
                    if current_debug:
                        print(f"  Marking cell ({row}, {col}) as OCCUPIED")
                else:
                    # Intermediate cells are free
                    self.grid[row, col] = self.FREE
                    if current_debug and j == len(cells) - 1:
                        print(f"  Marking cell ({row}, {col}) as FREE")
    
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
            
            # Create initial image with origin at top left (0,0)
            cmap = plt.cm.colors.ListedColormap(['gray', 'white', 'black'])
            bounds = [-0.5, 0.5, 1.5, 2.5]
            norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
            
            # Use origin='upper' to have (0,0) at top left
            self.map_img = self.ax.imshow(self.grid, cmap=cmap, norm=norm, origin='upper', 
                                         extent=[0, self.grid_width, self.grid_height, 0])
            
            # Add colorbar
            cbar = self.fig.colorbar(self.map_img, ticks=[0, 1, 2])
            cbar.ax.set_yticklabels(['Unseen', 'Free', 'Occupied'])
            
            # Set axis labels
            self.ax.set_xlabel('X (grid cells)')
            self.ax.set_ylabel('Y (grid cells)')
            self.ax.set_title('Occupancy Map (Origin at top left)')
            
            # Add grid lines
            self.ax.set_xticks(np.arange(0, self.grid_width + 1, 5))
            self.ax.set_yticks(np.arange(0, self.grid_height + 1, 5))
            self.ax.grid(color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
            
            # Add robot marker
            if self.last_robot_pos is not None:
                robot_row, robot_col = self.world_to_grid(*self.last_robot_pos)
                # For plotting with origin='upper', we use col for x and row for y
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
                    angle_inc = 2 * math.pi / self.n_sensors
                    
                    for i in range(self.n_sensors):
                        beam_angle = self.last_robot_angle + i * angle_inc
                        end_x = self.last_robot_pos[0] + MAX_SENSOR_DISTANCE * math.cos(beam_angle)
                        end_y = self.last_robot_pos[1] + MAX_SENSOR_DISTANCE * math.sin(beam_angle)
                        
                        end_row, end_col = self.world_to_grid(end_x, end_y)
                        line = self.ax.plot([robot_col, end_col], [robot_row, end_row], 'r-', alpha=0.3)[0]
                        self.lidar_lines.append(line)
            
            # Update grid lines to ensure they're visible
            self.ax.grid(color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
            
            self.fig.canvas.draw()
            plt.pause(0.001)
    
    def plot(self):
        """Plot the occupancy map."""
        self.update_plot()
        
    def visualize_mppi_trajectories(self, trajectories, chosen_trajectory=None):
        """
        Visualize the MPPI sampled trajectories and the chosen trajectory in continuous space.
        
        Args:
            trajectories (torch.Tensor): Tensor of shape (M*K, T, nx) containing sampled trajectories
            chosen_trajectory (torch.Tensor, optional): Tensor of shape (T, nx) containing the chosen trajectory
        """
        if self.fig is None:
            self.initialize_plot()
            
        # Create a separate figure for continuous space visualization
        if not hasattr(self, 'continuous_fig') or self.continuous_fig is None:
            self.continuous_fig, self.continuous_ax = plt.subplots(figsize=(8, 8))
            plt.ion()  # Enable interactive mode
            self.continuous_ax.set_xlabel('X (world units)')
            self.continuous_ax.set_ylabel('Y (world units)')
            self.continuous_ax.set_title('MPPI Trajectories (Continuous Space)')
            self.continuous_ax.set_xlim(0, self.width)
            self.continuous_ax.set_ylim(self.height, 0)  # Flip Y-axis to match grid coordinates (0,0 at top-left)
            self.continuous_ax.grid(True)
            
            # Add a colorbar for the occupancy map
            self.continuous_occupancy_img = self.continuous_ax.imshow(
                self.grid, 
                cmap=plt.cm.colors.ListedColormap(['gray', 'white', 'black']),
                norm=plt.cm.colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3),
                origin='upper',  # This is correct - origin at top-left
                extent=[0, self.width, self.height, 0],
                alpha=0.3  # Make it semi-transparent
            )
            
            # Initialize empty lists for trajectory lines
            self.continuous_mppi_lines = []
            self.continuous_chosen_line = None
            self.continuous_robot_marker = None
            
        # Clear previous trajectory lines in continuous space
        for line in self.continuous_mppi_lines:
            line.remove()
        self.continuous_mppi_lines = []
        
        if self.continuous_chosen_line is not None:
            self.continuous_chosen_line.remove()
            self.continuous_chosen_line = None
            
        # Update the occupancy map in the continuous space visualization
        self.continuous_occupancy_img.set_data(self.grid)
        
        # Update robot position in continuous space
        if self.last_robot_pos is not None:
            if self.continuous_robot_marker is None:
                self.continuous_robot_marker = self.continuous_ax.plot(
                    [self.last_robot_pos[0]], [self.last_robot_pos[1]], 
                    'ro', markersize=10, label='Robot Position'
                )[0]
            else:
                self.continuous_robot_marker.set_data(
                    [self.last_robot_pos[0]], [self.last_robot_pos[1]]
                )
                
            # Draw robot orientation as an arrow
            if hasattr(self, 'continuous_orientation_arrow') and self.continuous_orientation_arrow is not None:
                self.continuous_orientation_arrow.remove()
                
            if self.last_robot_angle is not None:
                arrow_length = 1.0  # Length of the orientation arrow
                dx = arrow_length * math.cos(self.last_robot_angle)
                # Flip the y-component to match the flipped y-axis
                dy = arrow_length * math.sin(self.last_robot_angle)
                self.continuous_orientation_arrow = self.continuous_ax.arrow(
                    self.last_robot_pos[0], self.last_robot_pos[1], dx, dy,
                    head_width=0.3, head_length=0.5, fc='r', ec='r'
                )
            
        # Convert trajectories to numpy for plotting in continuous space
        if trajectories is not None:
            trajectories_np = trajectories.detach().cpu().numpy()
            num_trajectories = trajectories_np.shape[0]
            
            # Plot a subset of trajectories to avoid cluttering the plot
            max_trajectories_to_plot = min(50, num_trajectories)
            step = max(1, num_trajectories // max_trajectories_to_plot)
            
            for i in range(0, num_trajectories, step):
                traj = trajectories_np[i]
                # Extract x and y coordinates directly (no grid conversion)
                traj_x = traj[:, 0]
                traj_y = traj[:, 1]
                
                # Plot the trajectory with low alpha to show density
                line = self.continuous_ax.plot(traj_x, traj_y, 'b-', alpha=0.1, linewidth=1)[0]
                self.continuous_mppi_lines.append(line)
        
        # Plot the chosen trajectory with a different color and higher alpha
        if chosen_trajectory is not None:
            chosen_traj_np = chosen_trajectory.detach().cpu().numpy()
            chosen_x = chosen_traj_np[:, 0]
            chosen_y = chosen_traj_np[:, 1]
            
            self.continuous_chosen_line = self.continuous_ax.plot(
                chosen_x, chosen_y, 'g-', alpha=0.8, linewidth=2, label='Chosen Trajectory'
            )[0]
            
            # Add goal marker
            if hasattr(self, 'goal_marker') and self.goal_marker is not None:
                self.goal_marker.remove()
                
            self.goal_marker = self.continuous_ax.plot(
                [robot_goal[0]], [robot_goal[1]], 
                'g*', markersize=15, label='Goal'
            )[0]
            
        # Add legend
        if not hasattr(self, 'legend_added') or not self.legend_added:
            self.continuous_ax.legend()
            self.legend_added = True
            
        self.continuous_fig.canvas.draw()
        plt.pause(0.001)
        
        # Also update the original grid visualization
        self.update_plot()


def main():

    env = posggym.make('DrivingContinuous-v0', world="30x30ScatteredObstacleField", num_agents=1, n_sensors=N_SENSORS, obs_dist=MAX_SENSOR_DISTANCE, render_mode="human")
    # env = posggym.make('DrivingContinuous-v0', world="30x30Empty", num_agents=1, n_sensors=N_SENSORS, obs_dist=MAX_SENSOR_DISTANCE, render_mode="human")


    # Comment out WarmStartSolver since we're focusing on MPPI visualization
    solver = WarmStartSolver(
        config=WarmStartSolverConfig(
            system_name="dubins3d",
            domain_cells=[int(MAP_WIDTH * MAP_RESOLUTION), int(MAP_HEIGHT * MAP_RESOLUTION), 40],
            domain=[[0, 0, 0], [MAP_WIDTH, MAP_HEIGHT, 2*np.pi]],
            mode="brt",
            accuracy="medium",
            converged_values=None,
            until_convergent=False,
            print_progress=False,
        )
    )


    occupancy_map = OccupancyMap(MAP_WIDTH, MAP_HEIGHT, MAP_RESOLUTION)

    observations, infos = env.reset()
    lidar_distances, vehicle_x, vehicle_y, vehicle_angle = observations["0"][0:N_SENSORS], observations["0"][2 * N_SENSORS], observations["0"][2 * N_SENSORS + 1], observations["0"][2*N_SENSORS + 2]

    nom_controller = Navigator()
    # Pass the actual grid dimensions, not the world dimensions
    grid_dimensions = [occupancy_map.grid_height, occupancy_map.grid_width]
    
    # Scale the origin and goal based on the resolution
    # The MPPI controller expects these in grid coordinates, not world coordinates
    scaled_origin = [ROBOT_ORIGIN[0] / MAP_RESOLUTION, ROBOT_ORIGIN[1] / MAP_RESOLUTION]
    global robot_goal
    robot_goal = env.state[0].dest_coord
    scaled_goal = [robot_goal[0] / MAP_RESOLUTION, robot_goal[1] / MAP_RESOLUTION]
    
    nom_controller.set_map(occupancy_map.grid != occupancy_map.FREE, grid_dimensions, scaled_origin, MAP_RESOLUTION)
    nom_controller.set_goal(scaled_goal)

    # creates a failure map with the given width, height and resolution.
    # unobserved cells are initialized as fail set.

    for _ in range(3000):
        observation = observations["0"]

        lidar_distances = observation[0:N_SENSORS]
        vehicle_x = observation[2 * N_SENSORS]
        vehicle_y = observation[2 * N_SENSORS + 1]
        vehicle_angle = observation[2 * N_SENSORS + 2]
        vehicle_x_velocity = observation[2 * N_SENSORS + 3]
        vehicle_y_velocity = observation[2 * N_SENSORS + 4]
        # Note: goal distance is now at indices 2*N_SENSORS + 5 and 2*N_SENSORS + 6
        env.render()


        print(f"{vehicle_x=}, {vehicle_y=}, {vehicle_angle=}, {vehicle_x_velocity=}, {vehicle_y_velocity=}")

        # update the fail set from the lidar observations. cells that are free are marked as safe.
        # assumes the lidar observations are equally spaced from 0 to 2*pi
        
        # Enable debug output for the first few iterations to diagnose issues
        debug_lidar = _ < 5  # Only debug the first 5 iterations
        
        occupancy_map.update_from_lidar(lidar_distances, vehicle_x, vehicle_y, vehicle_angle, debug=debug_lidar)
        occupancy_map.update_plot()  # Update the plot with new data

        fail_set = occupancy_map.grid != occupancy_map.FREE

        # compute a nominal action via MPPI
        nom_controller.set_map(occupancy_map.grid != occupancy_map.FREE, grid_dimensions, scaled_origin, MAP_RESOLUTION)
        nom_controller.set_odom((vehicle_x, vehicle_y), vehicle_angle)
        mppi_action = nom_controller.get_command().cpu().numpy()
        
        # Get and visualize MPPI trajectories
        sampled_trajectories = nom_controller.get_sampled_trajectories()
        chosen_trajectory = nom_controller.get_chosen_trajectory()
        
        # Debug prints
        print(f"MPPI action (linear_vel, angular_vel): {mppi_action}")
        
        # Convert MPPI action to environment action format
        # MPPI: [linear_vel, angular_vel]
        # Env: [dyaw, dvel]
        # 
        # For dyaw, we can use the angular velocity directly (dt=0.1 is assumed in the environment)
        # For dvel, we need to convert from absolute velocity to change in velocity
        # We'll use the current velocity from the observation
        current_vel = np.linalg.norm(observations["0"][3:5])  # Get current velocity magnitude
        
        # Due to how the environment combines velocities, we need to negate the velocity difference
        # The environment adds a new velocity component in the new heading direction
        # A negative dvel effectively reduces the overall velocity
        dvel = current_vel - mppi_action[0]  # Negated change in velocity
        
        # Create the environment action
        action = np.array([mppi_action[1] * 0.1, dvel])  # dyaw = angular_vel * dt, dvel
        
        print(f"Current velocity: {current_vel}")
        print(f"Converted env action (dyaw, dvel): {action}")
        
        if chosen_trajectory is not None and len(chosen_trajectory) > 1:
            # Calculate direction vector from current position to next position in trajectory
            current_pos = chosen_trajectory[0][:2].cpu().numpy()
            next_pos = chosen_trajectory[1][:2].cpu().numpy()
            traj_direction = next_pos - current_pos
            traj_angle = np.arctan2(traj_direction[1], traj_direction[0])
            print(f"Trajectory direction: {traj_direction}, angle: {traj_angle}")
            print(f"Current angle: {vehicle_angle}")
            
            # Calculate expected velocity based on trajectory
            expected_linear_vel = np.linalg.norm(traj_direction) / nom_controller.dt
            expected_angular_vel = (chosen_trajectory[1][2].item() - chosen_trajectory[0][2].item()) / nom_controller.dt
            print(f"Expected velocities - linear: {expected_linear_vel}, angular: {expected_angular_vel}")
        
        occupancy_map.visualize_mppi_trajectories(sampled_trajectories, chosen_trajectory)

        # # now compute HJ reachability
        # values = solver.solve(fail_set.T, target_time=-10.0, dt=0.1, epsilon=0.0001)
        if False and values is not None:
            safe_action, _, _ = solver.compute_safe_control(np.array([vehicle_x, vehicle_y, vehicle_angle]), action, action_bounds=np.array([[0.0, 5.0], [-4.0, 4.0]]), values=values)
        else:   
            safe_action = action


        actions = {"0": safe_action}
        observations, rewards, terminations, truncations, all_done, infos = env.step(actions)
        if all_done:
            observations, infos = env.reset()

    env.close()
    plt.ioff()  # Turn off interactive mode when done
    plt.show()  # Show the final plot


if __name__ == "__main__":
    main()
