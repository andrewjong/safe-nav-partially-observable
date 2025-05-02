"""
Modified version of racecar_warm_hj.py that uses Plotly for interactive visualization.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import ndimage

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the original file
from run.racecar_warm_hj import RacecarWarmHJRunner

class RacecarWarmHJPlotlyRunner(RacecarWarmHJRunner):
    """
    Modified version of RacecarWarmHJRunner that uses Plotly for interactive visualization.
    """
    
    def visualize_hj_level_set(
        self,
        solver,
        values,
        fail_set,
        vehicle_x,
        vehicle_y,
        vehicle_angle,
        vehicle_velocity,
        safety_intervening,
    ):
        """
        Visualize the HJ level set and vehicle position using Plotly for interactive visualization.

        Args:
            solver: The HJ solver
            values: The value function
            fail_set: The fail set
            vehicle_x (float): Current vehicle x position
            vehicle_y (float): Current vehicle y position
            vehicle_angle (float): Current vehicle angle
            vehicle_velocity (float): Current vehicle velocity
            safety_intervening (bool): Whether the safety filter is currently intervening
        """
        # Get the current slice of the value function at the current vehicle angle
        state = np.array([vehicle_x, vehicle_y, vehicle_angle, vehicle_velocity])

        state_ind = solver.state_to_grid(state)
        angle_index = state_ind[2]
        velocity_index = state_ind[3]
        value_slice = np.array(values[:, :, angle_index, velocity_index])
        current_state_value = values[*state_ind]
        print(f"Current state value: {current_state_value}")

        # Create coordinate meshgrid for plotting
        x = np.linspace(0, self.occupancy_map.width, self.occupancy_map.grid_width)
        y = np.linspace(0, self.occupancy_map.height, self.occupancy_map.grid_height)
        X, Y = np.meshgrid(x, y)

        # Create a new Plotly figure
        fig = make_subplots(rows=1, cols=1)

        # Plot the value function as a heatmap
        min_val, max_val = np.min(value_slice), np.max(value_slice)
        value_range = max_val - min_val
        vmin, vmax = min_val - 0.1 * value_range, max_val + 0.1 * value_range

        # Create a heatmap for the value function
        # Note: Plotly's heatmap expects z as a 2D array where z[i][j] corresponds to y[i] and x[j]
        value_heatmap = go.Heatmap(
            z=value_slice,
            x=x,
            y=y,
            colorscale='Viridis',
            zmin=vmin,
            zmax=vmax,
            colorbar=dict(title="Value Function"),
            hovertemplate='X: %{x:.2f}<br>Y: %{y:.2f}<br>Value: %{z:.4f}<extra></extra>',
            name="Value Function"
        )
        fig.add_trace(value_heatmap)

        # Plot the occupancy map
        # Convert occupancy map to a format suitable for Plotly
        occupancy_colors = {
            0: "black",  # Obstacle
            1: "white",  # Free space
            2: "gray"    # Unknown
        }
        
        # Create a list to store occupancy map traces
        for val, color in occupancy_colors.items():
            mask = self.occupancy_map.grid == val
            if np.any(mask):
                y_indices, x_indices = np.where(mask)
                for i in range(len(y_indices)):
                    y_idx, x_idx = y_indices[i], x_indices[i]
                    x_pos, y_pos = X[y_idx, x_idx], Y[y_idx, x_idx]
                    x_min, x_max = x_pos - 0.5 * (x[1] - x[0]), x_pos + 0.5 * (x[1] - x[0])
                    y_min, y_max = y_pos - 0.5 * (y[1] - y[0]), y_pos + 0.5 * (y[1] - y[0])
                    
                    # Only add obstacles as visible rectangles
                    if val == 0:  # Obstacle
                        fig.add_shape(
                            type="rect",
                            x0=x_min, y0=y_min, x1=x_max, y1=y_max,
                            line=dict(color="black", width=1),
                            fillcolor="black",
                            opacity=0.3,
                            layer="below"
                        )

        # Plot the unsafe boundary
        try:
            # Create a binary mask for unsafe cells (value <= 0)
            unsafe_mask = value_slice <= 0
            
            # Find the boundary cells
            struct = ndimage.generate_binary_structure(2, 1)
            eroded = ndimage.binary_erosion(unsafe_mask, structure=struct)
            boundary = unsafe_mask & ~eroded
            
            # Get the coordinates of boundary cells
            boundary_y, boundary_x = np.where(boundary)
            
            # Draw the boundary cells with lines
            for i in range(len(boundary_y)):
                y_idx, x_idx = boundary_y[i], boundary_x[i]
                x_pos, y_pos = X[y_idx, x_idx], Y[y_idx, x_idx]
                x_min, x_max = x_pos - 0.5 * (x[1] - x[0]), x_pos + 0.5 * (x[1] - x[0])
                y_min, y_max = y_pos - 0.5 * (y[1] - y[0]), y_pos + 0.5 * (y[1] - y[0])
                
                # Add a rectangle shape for each boundary cell
                fig.add_shape(
                    type="rect",
                    x0=x_min, y0=y_min, x1=x_max, y1=y_max,
                    line=dict(color="red", width=2),
                    fillcolor="rgba(0,0,0,0)",
                    layer="above"
                )
            
            # Add a text annotation for the unsafe region
            if np.any(unsafe_mask):
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
                        fig.add_annotation(
                            x=X[center_y, center_x],
                            y=Y[center_y, center_x],
                            text="Unsafe",
                            showarrow=False,
                            font=dict(color="red", size=14, family="Arial Black"),
                        )
        except Exception as e:
            print(f"Could not plot unsafe boundary: {e}")

        # Plot the fail set boundary (commented out for now as in the original code)
        # This can be uncommented and adapted similar to the unsafe boundary if needed

        # Plot the vehicle position with color based on safety intervention
        robot_color = "cyan" if safety_intervening else "red"
        robot_label = (
            "Vehicle (Safety Active)" if safety_intervening else "Vehicle (Nominal)"
        )
        
        # Add the vehicle as a marker
        fig.add_trace(go.Scatter(
            x=[vehicle_x],
            y=[vehicle_y],
            mode='markers',
            marker=dict(
                color=robot_color,
                size=15,
                symbol='circle'
            ),
            name=robot_label
        ))
        
        # Add an arrow to show vehicle orientation
        arrow_length = 1.0
        dx = arrow_length * np.cos(vehicle_angle)
        dy = arrow_length * np.sin(vehicle_angle)
        
        # Create the arrow as a line with an arrowhead
        fig.add_trace(go.Scatter(
            x=[vehicle_x, vehicle_x + dx],
            y=[vehicle_y, vehicle_y + dy],
            mode='lines',
            line=dict(color=robot_color, width=3),
            showlegend=False
        ))
        
        # Add arrowhead
        head_length = 0.5
        head_width = 0.3
        
        # Calculate arrowhead points
        arrow_angle = np.arctan2(dy, dx)
        arrow_head_x = vehicle_x + dx - head_length * np.cos(arrow_angle)
        arrow_head_y = vehicle_y + dy - head_length * np.sin(arrow_angle)
        
        left_angle = arrow_angle + np.pi/2
        right_angle = arrow_angle - np.pi/2
        
        left_x = arrow_head_x + head_width/2 * np.cos(left_angle)
        left_y = arrow_head_y + head_width/2 * np.sin(left_angle)
        
        right_x = arrow_head_x + head_width/2 * np.cos(right_angle)
        right_y = arrow_head_y + head_width/2 * np.sin(right_angle)
        
        # Add arrowhead as a filled triangle
        fig.add_trace(go.Scatter(
            x=[vehicle_x + dx, left_x, right_x, vehicle_x + dx],
            y=[vehicle_y + dy, left_y, right_y, vehicle_y + dy],
            fill="toself",
            fillcolor=robot_color,
            line=dict(color=robot_color),
            showlegend=False
        ))

        # Add timestamp
        timestamp = f"Time: {time.time():.1f}s"
        fig.add_annotation(
            x=0.02,
            y=0.02,
            xref="paper",
            yref="paper",
            text=timestamp,
            showarrow=False,
            font=dict(size=10),
            bgcolor="rgba(255, 255, 255, 0.5)",
            bordercolor="rgba(0, 0, 0, 0.5)",
            borderwidth=1,
            borderpad=4,
            align="left"
        )

        # Set plot title and labels
        fig.update_layout(
            title="HJ Reachability Level Set Visualization",
            xaxis_title="X Position",
            yaxis_title="Y Position",
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            ),
            # Set equal aspect ratio
            yaxis=dict(
                scaleanchor="x",
                scaleratio=1,
                autorange="reversed"  # Invert y-axis to match matplotlib
            ),
            # Make the plot more interactive
            hovermode='closest',
            # Set the size of the figure
            width=1000,
            height=800,
        )

        # Set axis limits to match the occupancy map dimensions
        fig.update_xaxes(range=[0, self.occupancy_map.width])
        fig.update_yaxes(range=[0, self.occupancy_map.height])

        # Store the figure for later reference
        self.hj_fig = fig
        
        # Show the figure
        fig.show()


# Main function to run the simulation
def main():
    """
    Main function to run the simulation with Plotly visualization.
    """
    # Create the runner
    runner = RacecarWarmHJPlotlyRunner()
    
    # Run the simulation
    runner.run()


if __name__ == "__main__":
    main()