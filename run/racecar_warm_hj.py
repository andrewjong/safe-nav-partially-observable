#!/usr/bin/env python3
"""
Racecar Hamilton-Jacobi Reachability with Warm Start

This script implements a safety filter for a racecar using Hamilton-Jacobi (HJ) reachability
analysis with warm start. It uses MPPI (Model Predictive Path Integral) for nominal control
and HJ reachability for safety guarantees.
"""

# Standard library imports
import math
import time

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import posggym

# Local imports
from reachability.warm_start_solver import WarmStartSolver, WarmStartSolverConfig
from src.mppi import Navigator

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

# Environment setup
MAP_RESOLUTION = 0.25  # units per cell
N_SENSORS = 128
MAX_SENSOR_DISTANCE = 50.0
MAX_SENSOR_DISTANCE = 5.0
ROBOT_ORIGIN = [1, 1]
FOV = np.pi / 4  # 45-degree view centered at the front of the agent
FOV = np.pi / 3  # 45-degree view centered at the front of the agent
FOV = np.pi * 2

MARK_FREE_RADIUS = 3.0

# BRT (Backward Reachable Tube) parameters
# https://posggym.readthedocs.io/en/latest/environments/continuous/driving_continuous.html#state-space
THETA_MIN = 0
THETA_MAX = 2 * np.pi + 1.0  # add an epsilon to avoid numerical issues
THETA_NUM_CELLS = 13
VELOCITY_MIN = -1.41 # Minimum velocity
VELOCITY_MAX = 1.41 + 0.1  # add an epsilon to avoid numerical issues
VELOCITY_NUM_CELLS = 21

# Cell state constants
UNSEEN = 0
FREE = 1
OCCUPIED = 2

# -----------------------------------------------------------------------------
# Classes
# -----------------------------------------------------------------------------

class OccupancyMap:
    """
    A class to build and maintain an occupancy grid map from lidar observations.

    The map is represented as a 2D grid where each cell can be in one of three states:
    - UNSEEN: Cell has not been observed yet
    - FREE: Cell has been observed and is free (no obstacle)
    - OCCUPIED: Cell has been observed and contains an obstacle
    """

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

        # Last known robot position for visualization
        self.last_robot_pos = None
        self.last_robot_angle = None
        self.n_sensors = None

    def reset(self):
        """Reset the occupancy map to its initial state."""
        self.grid.fill(UNSEEN)
        self.last_robot_pos = None
        self.last_robot_angle = None

    def mark_free_radius(self, robot_x, robot_y, radius):
        """
        Mark cells within a specified radius of the robot's position as FREE.
        
        This is useful for initializing the map with known free space around
        the robot's starting position.
        
        Args:
            robot_x (float): Robot's x-coordinate in world units
            robot_y (float): Robot's y-coordinate in world units
            radius (float): Radius around the robot to mark as free, in world units
        """
        # Convert radius to grid units
        radius_grid = int(np.ceil(radius / self.resolution))
        
        # Get the robot's position in grid coordinates
        center_row, center_col = self.world_to_grid(robot_x, robot_y)
        
        # Determine the bounds of the circle in grid coordinates
        min_row = max(0, center_row - radius_grid)
        max_row = min(self.grid_height - 1, center_row + radius_grid)
        min_col = max(0, center_col - radius_grid)
        max_col = min(self.grid_width - 1, center_col + radius_grid)
        
        # Mark all cells within the radius as FREE
        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                # Calculate squared distance from center
                dr = row - center_row
                dc = col - center_col
                dist_squared = dr * dr + dc * dc
                
                # If the cell is within the radius, mark it as FREE
                if dist_squared <= radius_grid * radius_grid:
                    self.grid[row, col] = FREE

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

    def update_from_lidar(self, lidar_distances, vehicle_x, vehicle_y, vehicle_angle, fov=2*math.pi, debug=False):
        """
        Update the occupancy map based on lidar observations.

        Args:
            lidar_distances (numpy.ndarray): Array of lidar distance readings
            vehicle_x (float): X coordinate of the vehicle in world units
            vehicle_y (float): Y coordinate of the vehicle in world units
            vehicle_angle (float): Orientation of the vehicle in radians
            fov (float): Field of view in radians (default: 2*pi for 360 degrees)
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

        # Calculate angle bounds
        angle_min = -fov / 2
        angle_max = fov / 2

        # Angle between consecutive lidar beams
        angle_inc = fov / self.n_sensors

        # Process each lidar beam
        for i, distance in enumerate(lidar_distances):
            # Only process every 4th beam in debug mode to reduce output
            current_debug = debug and (i % 4 == 0)

            # Calculate beam angle in world frame
            # Map i from [0, n_sensors-1] to [angle_min, angle_max]
            beam_angle = vehicle_angle + angle_min + i * angle_inc

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
                    self.grid[row, col] = OCCUPIED
                    if current_debug:
                        print(f"  Marking cell ({row}, {col}) as OCCUPIED")
                else:
                    # Intermediate cells are free
                    self.grid[row, col] = FREE
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


class MapVisualizer:
    """
    A class to handle visualization of the occupancy map, MPPI trajectories, and HJ level sets.
    """

    def __init__(self, occupancy_map):
        """
        Initialize the visualizer.

        Args:
            occupancy_map (OccupancyMap): The occupancy map to visualize
        """
        self.occupancy_map = occupancy_map
        
        # MPPI visualization variables
        self.mppi_fig = None
        self.mppi_ax = None
        self.occupancy_img = None
        self.mppi_lines = []
        self.chosen_line = None
        self.robot_marker = None
        self.orientation_arrow = None
        self.goal_marker = None
        self.legend_added = False
        
        # HJ visualization variables
        self.hj_fig = None

    def reset(self):
        """Reset the visualization to its initial state."""
        # Reset MPPI figure elements without closing the window
        if self.mppi_fig is not None:
            # Clear previous trajectory lines
            for line in self.mppi_lines:
                if line in self.mppi_ax.lines:
                    line.remove()
            self.mppi_lines = []

            # Clear chosen trajectory line
            if self.chosen_line is not None and self.chosen_line in self.mppi_ax.lines:
                self.chosen_line.remove()
                self.chosen_line = None

            # Clear robot marker and orientation arrow
            if self.robot_marker is not None and self.robot_marker in self.mppi_ax.lines:
                self.robot_marker.remove()
                self.robot_marker = None

            if self.orientation_arrow is not None:
                self.orientation_arrow.remove()
                self.orientation_arrow = None

            # Clear goal marker
            if self.goal_marker is not None and self.goal_marker in self.mppi_ax.lines:
                self.goal_marker.remove()
                self.goal_marker = None

            # Reset the grid data
            if self.occupancy_img is not None:
                self.occupancy_img.set_data(self.occupancy_map.grid)

            # Reset legend state
            self.legend_added = False

            # Redraw the figure
            if self.mppi_fig.canvas is not None:
                self.mppi_fig.canvas.draw_idle()
                plt.pause(0.001)

    def visualize_mppi_trajectories(self, trajectories, chosen_trajectory=None, robot_goal=None):
        """
        Visualize the MPPI sampled trajectories and the chosen trajectory in continuous space.

        Args:
            trajectories (torch.Tensor): Tensor of shape (M*K, T, nx) containing sampled trajectories
            chosen_trajectory (torch.Tensor, optional): Tensor of shape (T, nx) containing the chosen trajectory
            robot_goal (tuple, optional): The goal position (x, y) for the robot
        """
        # Create figure for visualization with occupancy map and MPPI trajectories if it doesn't exist
        if self.mppi_fig is None:
            # Create a new figure with a unique number to avoid conflicts
            self.mppi_fig, self.mppi_ax = plt.subplots(
                figsize=(8, 8), num="MPPI Visualization"
            )
            plt.ion()  # Enable interactive mode

            # Set up the axes
            self.mppi_ax.set_xlabel("X (world units)")
            self.mppi_ax.set_ylabel("Y (world units)")
            self.mppi_ax.set_title("Occupancy Map with MPPI Trajectories")
            self.mppi_ax.set_xlim(0, self.occupancy_map.width)
            self.mppi_ax.set_ylim(
                self.occupancy_map.height, 0
            )  # Flip Y-axis to match grid coordinates (0,0 at top-left)
            self.mppi_ax.grid(True)

            # Add a colorbar for the occupancy map
            self.occupancy_img = self.mppi_ax.imshow(
                self.occupancy_map.grid,
                cmap=plt.cm.colors.ListedColormap(["gray", "white", "black"]),
                norm=plt.cm.colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3),
                origin="upper",  # This is correct - origin at top-left
                extent=[0, self.occupancy_map.width, self.occupancy_map.height, 0],
                alpha=0.5,  # Make it semi-transparent
            )

            # Initialize empty lists for trajectory lines
            self.mppi_lines = []
            self.chosen_line = None
            self.robot_marker = None

            # Make sure the figure is visible
            self.mppi_fig.canvas.draw()
            plt.pause(0.001)

        # Clear previous trajectory lines in continuous space
        for line in self.mppi_lines:
            if line in self.mppi_ax.lines:
                line.remove()
        self.mppi_lines = []

        if self.chosen_line is not None and self.chosen_line in self.mppi_ax.lines:
            self.chosen_line.remove()
            self.chosen_line = None

        # Update the occupancy map in the continuous space visualization
        if self.occupancy_img is not None:
            self.occupancy_img.set_data(self.occupancy_map.grid)

        # Update robot position in continuous space
        if self.occupancy_map.last_robot_pos is not None:
            # Update or create robot marker
            if self.robot_marker is None:
                self.robot_marker = self.mppi_ax.plot(
                    [self.occupancy_map.last_robot_pos[0]],
                    [self.occupancy_map.last_robot_pos[1]],
                    "ro",
                    markersize=10,
                    label="Robot Position",
                )[0]
            elif self.robot_marker in self.mppi_ax.lines:
                self.robot_marker.set_data(
                    [self.occupancy_map.last_robot_pos[0]], [self.occupancy_map.last_robot_pos[1]]
                )
            else:
                # If the marker was removed, create a new one
                self.robot_marker = self.mppi_ax.plot(
                    [self.occupancy_map.last_robot_pos[0]],
                    [self.occupancy_map.last_robot_pos[1]],
                    "ro",
                    markersize=10,
                    label="Robot Position",
                )[0]

            # Draw robot orientation as an arrow
            if self.orientation_arrow is not None:
                try:
                    self.orientation_arrow.remove()
                except:
                    # Arrow might have been removed already
                    pass

            if self.occupancy_map.last_robot_angle is not None:
                arrow_length = 1.0  # Length of the orientation arrow
                dx = arrow_length * math.cos(self.occupancy_map.last_robot_angle)
                # Flip the y-component to match the flipped y-axis
                dy = arrow_length * math.sin(self.occupancy_map.last_robot_angle)
                self.orientation_arrow = self.mppi_ax.arrow(
                    self.occupancy_map.last_robot_pos[0],
                    self.occupancy_map.last_robot_pos[1],
                    dx,
                    dy,
                    head_width=0.3,
                    head_length=0.5,
                    fc="r",
                    ec="r",
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
                line = self.mppi_ax.plot(
                    traj_x, traj_y, "b-", alpha=0.1, linewidth=1
                )[0]
                self.mppi_lines.append(line)

        # Plot the chosen trajectory with a different color and higher alpha
        if chosen_trajectory is not None:
            chosen_traj_np = chosen_trajectory.detach().cpu().numpy()
            chosen_x = chosen_traj_np[:, 0]
            chosen_y = chosen_traj_np[:, 1]

            self.chosen_line = self.mppi_ax.plot(
                chosen_x,
                chosen_y,
                "g-",
                alpha=0.8,
                linewidth=2,
                label="Chosen Trajectory",
            )[0]

            # Add goal marker
            if self.goal_marker is not None:
                try:
                    if self.goal_marker in self.mppi_ax.lines:
                        self.goal_marker.remove()
                except:
                    # Goal marker might have been removed already
                    pass

            if robot_goal is not None:
                self.goal_marker = self.mppi_ax.plot(
                    [robot_goal[0]], [robot_goal[1]], "g*", markersize=15, label="Goal"
                )[0]

        # Add or update legend
        self.mppi_ax.legend()
        self.legend_added = True

        # Redraw the figure without closing it
        if self.mppi_fig.canvas is not None:
            self.mppi_fig.canvas.draw_idle()
            plt.pause(0.001)

    def visualize_hj_level_set(
        self,
        solver: WarmStartSolver,
        values: np.ndarray,
        fail_set: np.ndarray,
        vehicle_x: float,
        vehicle_y: float,
        vehicle_angle: float,
        vehicle_velocity: float,
        safety_intervening=False,
    ):
        """
        Visualize the HJ reachability level set on a separate heat map plot.

        Args:
            values (np.ndarray): The value function from HJ reachability computation
            fail_set (np.ndarray): The fail set (obstacles)
            vehicle_x (float): Current vehicle x position
            vehicle_y (float): Current vehicle y position
            vehicle_angle (float): Current vehicle orientation
            vehicle_velocity (float): Current vehicle velocity
            safety_intervening (bool): Whether the safety filter is currently intervening
        """
        # Check if we already have a figure for HJ visualization
        if self.hj_fig is None:
            self.hj_fig = plt.figure(figsize=(10, 8), num="HJ Visualization")
            self.hj_fig.hj_visualization = True
        else:
            plt.figure(self.hj_fig.number)
            plt.clf()

        ax = self.hj_fig.add_subplot(111)

        # Get the current slice of the value function at the current vehicle angle
        state = np.array([vehicle_x, vehicle_y, vehicle_angle, vehicle_velocity])
        print(f"{state=}")

        state_ind = solver.state_to_grid(state)
        angle_index = state_ind[2]
        velocity_index = state_ind[3]
        # velocity_index = -1
        print(f"{state_ind=}")
        value_slice = np.array(values[:, :, angle_index, velocity_index])
        current_state_value = values[*state_ind]
        print(f"Current state value: {current_state_value}")

        # Create coordinate meshgrid for plotting
        x = np.linspace(0, self.occupancy_map.width, self.occupancy_map.grid_width)
        y = np.linspace(0, self.occupancy_map.height, self.occupancy_map.grid_height)
        X, Y = np.meshgrid(x, y)

        # Plot the occupancy map as background
        occupancy_cmap = plt.cm.colors.ListedColormap(["black", "white", "gray"])
        occupancy_bounds = [-0.5, 0.5, 1.5, 2.5]
        occupancy_norm = plt.cm.colors.BoundaryNorm(occupancy_bounds, occupancy_cmap.N)
        ax.pcolormesh(
            X, Y, self.occupancy_map.grid, cmap=occupancy_cmap, norm=occupancy_norm, alpha=0.3
        )

        # Plot the value function as a discrete heat map
        # Clip the values to a reasonable range for better visualization
        min_val, max_val = np.min(value_slice), np.max(value_slice)
        value_range = max_val - min_val
        vmin, vmax = min_val - 0.1 * value_range, max_val + 0.1 * value_range
        
        # Comment out the contourf visualization
        # value_contour = ax.contourf(
        #     X, Y, value_slice, levels=20, cmap="viridis", alpha=0.7, vmin=vmin, vmax=vmax
        # )
        
        # Instead, use pcolormesh for a discrete cell-by-cell visualization
        value_heatmap = ax.pcolormesh(
            X, Y, value_slice, cmap="viridis", alpha=0.7, vmin=vmin, vmax=vmax, 
            edgecolors='face', shading='auto'
        )
        cbar = plt.colorbar(value_heatmap, ax=ax, label="Value Function")

        # Plot the unsafe boundary by tracing cell boundaries
        try:
            # Create a binary mask for unsafe cells (value <= 0)
            unsafe_mask = value_slice <= 0
            
            # Create a new array with NaN for safe cells and 1 for unsafe cells
            unsafe_display = np.where(unsafe_mask, 1, np.nan)
            
            # Plot the unsafe cells with a red color and visible cell edges
            unsafe_boundary = ax.pcolormesh(
                X, Y, unsafe_display, 
                cmap=plt.cm.colors.ListedColormap(['red']),
                alpha=0.5,
                edgecolors='none',
                linewidths=1.5,
                shading='auto'
            )
            
            # Add a text annotation for the unsafe region
            # Find the center of the largest unsafe region
            if np.any(unsafe_mask):
                from scipy import ndimage
                # Label connected regions
                labeled_mask, num_features = ndimage.label(unsafe_mask)
                if num_features > 0:
                    # Find the largest region
                    largest_region = np.argmax(np.bincount(labeled_mask.flat)[1:]) + 1
                    # Get coordinates of the largest region
                    y_indices, x_indices = np.where(labeled_mask == largest_region)
                    if len(y_indices) > 0:
                        # Calculate center of the region
                        center_y = int(np.mean(y_indices))
                        center_x = int(np.mean(x_indices))
                        # Add text annotation
                        ax.text(X[center_y, center_x], Y[center_y, center_x], 
                                "Unsafe", color='white', fontweight='bold',
                                ha='center', va='center')
        except Exception as e:
            print(f"Could not plot unsafe boundary: {e}")

        # Plot the fail set boundary by tracing cell boundaries
        try:
            # Convert fail_set to numpy array if it's not already
            fail_set_np = np.array(fail_set)
            
            # Create a binary mask for fail cells
            fail_mask = fail_set_np > 0.5
            
            # Create a new array with NaN for non-fail cells and 1 for fail cells
            fail_display = np.where(fail_mask, 1, np.nan)
            
            # Plot the fail cells with a black color and visible cell edges
            fail_boundary = ax.pcolormesh(
                X, Y, fail_display, 
                cmap=plt.cm.colors.ListedColormap(['black']),
                alpha=0.5,
                edgecolors='none',
                linewidths=1.5,
                shading='auto'
            )
            
            # Add a text annotation for the fail region
            if np.any(fail_mask):
                from scipy import ndimage
                # Label connected regions
                labeled_mask, num_features = ndimage.label(fail_mask)
                if num_features > 0:
                    # Find the largest region
                    largest_region = np.argmax(np.bincount(labeled_mask.flat)[1:]) + 1
                    # Get coordinates of the largest region
                    y_indices, x_indices = np.where(labeled_mask == largest_region)
                    if len(y_indices) > 0:
                        # Calculate center of the region
                        center_y = int(np.mean(y_indices))
                        center_x = int(np.mean(x_indices))
                        # Add text annotation
                        ax.text(X[center_y, center_x], Y[center_y, center_x], 
                                "Fail", color='white', fontweight='bold',
                                ha='center', va='center')
        except Exception as e:
            print(f"Could not plot fail set boundary: {e}")

        # Plot the vehicle position with color based on safety intervention
        robot_color = "cyan" if safety_intervening else "red"
        robot_label = (
            "Vehicle (Safety Active)" if safety_intervening else "Vehicle (Nominal)"
        )
        ax.plot(
            vehicle_x, vehicle_y, "o", color=robot_color, markersize=10, label=robot_label
        )

        # Add an arrow to show vehicle orientation
        arrow_length = 1.0
        dx = arrow_length * np.cos(vehicle_angle)
        dy = arrow_length * np.sin(vehicle_angle)
        ax.arrow(
            vehicle_x,
            vehicle_y,
            dx,
            dy,
            head_width=0.3,
            head_length=0.5,
            fc=robot_color,
            ec=robot_color,
        )

        # Set plot title and labels
        ax.set_title("HJ Reachability Level Set Visualization")
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")

        # Add legend
        ax.legend(loc="upper right")

        # Set equal aspect ratio
        ax.set_aspect("equal")

        # Invert y-axis to have (0,0) in the top left
        ax.invert_yaxis()

        # Set axis limits to match the occupancy map dimensions
        ax.set_xlim(0, self.occupancy_map.width)
        ax.set_ylim(self.occupancy_map.height, 0)  # Inverted y-axis

        # Add timestamp and safety status
        timestamp = f"Time: {time.time():.1f}s"

        # Add timestamp at bottom left
        ax.text(
            0.02,
            0.02,
            timestamp,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
        )

        # Show the plot without blocking
        self.hj_fig.canvas.draw_idle()
        plt.pause(0.001)


# -----------------------------------------------------------------------------
# Main function
# -----------------------------------------------------------------------------

def main():
    """Main function to run the simulation."""
    # Create the environment
    global env, robot_goal
    robot_goal = None
    
    env = posggym.make(
        "DrivingContinuous-v0",
        # world="30x30OneWallDiagonal",
        # world="30x30EmptyStraight",
        # world="14x14Empty",
        world="14x14OneWall",
        # world="30x30Empty",
        # world="30x30ScatteredObstacleField",
        # world="14x14Sparse",
        num_agents=1,
        n_sensors=N_SENSORS,
        obs_dist=MAX_SENSOR_DISTANCE,
        fov=FOV,
        render_mode="human",
    )

    # Get map dimensions from environment
    map_width = env.model.state_space[0][0].high[0]
    map_height = env.model.state_space[0][0].high[1]

    # Initialize HJ reachability solver
    config = WarmStartSolverConfig(
        system_name="dubins3d_velocity",
        domain_cells=[
            int(map_width / MAP_RESOLUTION),
            int(map_height / MAP_RESOLUTION),
            THETA_NUM_CELLS,
            VELOCITY_NUM_CELLS
        ],
        domain=np.array([
            [0, 0, THETA_MIN, VELOCITY_MIN], 
            [map_width, map_height, THETA_MAX, VELOCITY_MAX]
        ]),
        mode="brt",
        accuracy="medium",
        converged_values=None,
        until_convergent=True,
        print_progress=False,
    )
    solver = WarmStartSolver(config=config)

    # Initialize occupancy map and visualizer
    occupancy_map = OccupancyMap(map_width, map_height, MAP_RESOLUTION)
    visualizer = MapVisualizer(occupancy_map)

    # Reset environment and get initial observations
    observations, infos = env.reset()
    lidar_distances, vehicle_x, vehicle_y, vehicle_angle = (
        observations["0"][0:N_SENSORS],
        observations["0"][2 * N_SENSORS],
        observations["0"][2 * N_SENSORS + 1],
        observations["0"][2 * N_SENSORS + 2],
    )
    
    # Mark initial free space around the robot
    occupancy_map.mark_free_radius(vehicle_x, vehicle_y, MARK_FREE_RADIUS)

    # Initialize Navigator with the agent radius from the environment
    nom_controller = Navigator(robot_radius=env.model.world.agent_radius)
    
    # Main simulation loop
    for _ in range(3000):
        # Get current observation
        observation = observations["0"]
        lidar_distances = observation[0:N_SENSORS]
        vehicle_x = observation[2 * N_SENSORS]
        vehicle_y = observation[2 * N_SENSORS + 1]
        vehicle_angle = observation[2 * N_SENSORS + 2]
        vehicle_x_velocity = observation[2 * N_SENSORS + 3]
        vehicle_y_velocity = observation[2 * N_SENSORS + 4]
        
        # Calculate current velocity magnitude
        current_vel = np.linalg.norm(np.array([vehicle_x_velocity, vehicle_y_velocity]))
        
        # Render environment
        env.render()

        # Update occupancy map from lidar observations
        occupancy_map.update_from_lidar(lidar_distances, vehicle_x, vehicle_y, vehicle_angle, fov=FOV)

        # Get initial safe set (free cells)
        initial_safe_set = occupancy_map.grid == FREE

        # Set up MPPI controller
        scaled_origin = [ROBOT_ORIGIN[0] / MAP_RESOLUTION, ROBOT_ORIGIN[1] / MAP_RESOLUTION]
        robot_goal = env.state[0].dest_coord
        
        # Configure MPPI controller
        nom_controller.set_goal(robot_goal)
        nom_controller.set_map(
            occupancy_map.grid != FREE,  # Obstacle map (not free = obstacle)
            [occupancy_map.grid_height, occupancy_map.grid_width],  # Grid dimensions
            scaled_origin,  # Origin
            MAP_RESOLUTION,  # Resolution
        )
        nom_controller.set_state((vehicle_x, vehicle_y), vehicle_angle, current_vel)
        
        # Get nominal action from MPPI
        mppi_action = nom_controller.get_command().cpu().numpy()

        # Get and visualize MPPI trajectories
        sampled_trajectories = nom_controller.get_sampled_trajectories()
        chosen_trajectory = nom_controller.get_chosen_trajectory()

        # Visualize MPPI trajectories
        visualizer.visualize_mppi_trajectories(sampled_trajectories, chosen_trajectory, robot_goal)

        # Compute HJ reachability
        values = solver.solve(
            initial_safe_set, MAP_RESOLUTION, target_time=-10.0, dt=0.1, epsilon=0.0001
        )

        # Compute safe action if values are available
        if values is not None:
            # Compute safe action
            safe_mppi_action, _, _, has_intervened = solver.compute_safe_control(
                np.array([vehicle_x, vehicle_y, vehicle_angle, current_vel]),
                mppi_action,
                action_bounds=np.array([[-np.pi/4 * 0, np.pi/4 * 0], [-0.25, 0.25]]),
                values=values,
            )

            # Visualize HJ level set
            fail_set = np.logical_not(initial_safe_set)
            visualizer.visualize_hj_level_set(
                solver,
                values,
                fail_set,
                vehicle_x,
                vehicle_y,
                vehicle_angle,
                current_vel,
                safety_intervening=has_intervened,
            )
        else:
            safe_mppi_action = mppi_action
            has_intervened = False

        # Take a step in the environment
        observations, rewards, terminations, truncations, all_done, infos = env.step(
            {"0": safe_mppi_action}
        )
        
        # Check for collision
        reward = rewards["0"]
        if reward < 0:
            print("AGENT COLLIDED.")
            
        # Reset if episode is done
        if all_done:
            observations, infos = env.reset()
            occupancy_map.reset()
            visualizer.reset()
            
            # Get initial observations after reset
            lidar_distances, vehicle_x, vehicle_y, vehicle_angle = (
                observations["0"][0:N_SENSORS],
                observations["0"][2 * N_SENSORS],
                observations["0"][2 * N_SENSORS + 1],
                observations["0"][2 * N_SENSORS + 2],
            )
            
            # Mark initial free space
            occupancy_map.mark_free_radius(vehicle_x, vehicle_y, MARK_FREE_RADIUS)
            
            # Reinitialize solver
            solver = WarmStartSolver(config=config)

    # Clean up
    env.close()
    plt.ioff()  # Turn off interactive mode when done
    plt.show()  # Show the final plot


if __name__ == "__main__":
    main()