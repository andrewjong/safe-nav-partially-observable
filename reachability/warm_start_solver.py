import numpy as np
import skfmm
import cvxpy as cp

import hj_reachability as hj
import jax.numpy as jnp
from hj_reachability import dynamics, sets

from matplotlib import pyplot as plt
from dataclasses import dataclass
from time import time as time_pkg

import matplotlib.pyplot as plt


@dataclass
class WarmStartSolverConfig:
    system_name: str  # "dubins3d"
    # 1-dim: e.g [x_resolution, y_resolution, theta_resolution, v_resolution]
    domain_cells: np.ndarray
    # 2-dim: e.g [[x_min, y_min, theta_min, v_min], [x_max, y_max, theta_max, v_max]]
    domain: np.ndarray
    mode: str  # "brs" or "brt"
    accuracy: str  # "low", "medium", "high", "very_high"
    superlevel_set_epsilon: float = 0.1
    converged_values: np.ndarray | None = None
    until_convergent: bool = True
    print_progress: bool = True


class Dubins3DVelocity(dynamics.ControlAndDisturbanceAffineDynamics):
    def __init__(
        self,
        min_angular_velocity=-np.pi / 4,
        max_angular_velocity=np.pi / 4,
        min_linear_acceleration=-0.25,
        max_linear_acceleration=0.25,
        control_mode="max",
        disturbance_mode="min",
        control_space=None,
        disturbance_space=None,
    ):
        if control_space is None:
            control_space = sets.Box(
                jnp.array([min_angular_velocity, min_linear_acceleration]),
                jnp.array([max_angular_velocity, max_linear_acceleration]),
            )
        if disturbance_space is None:
            disturbance_space = sets.Box(jnp.array([0]), jnp.array([0]))
        super().__init__(
            control_mode, disturbance_mode, control_space, disturbance_space
        )

    def open_loop_dynamics(self, state, time):
        x, y, psi, v = state  # State now includes velocity v
        return jnp.array([v * jnp.cos(psi), v * jnp.sin(psi), 0.0, 0.0])

    def control_jacobian(self, state, time):
        x, y, psi, v = state  # Updated to unpack 4 state variables
        return jnp.array(
            [
                [0, 0],
                [0, 0],
                [1, 0],
                [0, 1],
            ]
        )

    def disturbance_jacobian(self, state, time):
        return jnp.array(
            [
                [0.0],
                [0.0],
                [0.0],
                [0.0],  # no disturbance
            ]
        )


class WarmStartSolver:
    def __init__(
        self,
        config: WarmStartSolverConfig,
    ):
        self.config = config

        self.problem_definition = None
        self.initial_values = (
            None  # used to check if robot reached a failure (not just unsafe) state
        )
        self.last_values = self.config.converged_values if not None else None
        self.last_grid_map = None

    def get_problem_definition(self, system, domain, dims, mode, accuracy):
        dynamics = Dubins3DVelocity()
        solver_settings = self.get_solver_settings(accuracy=accuracy, mode=mode)
        grid = self.get_domain_grid(domain, dims)
        problem_definition = {
            "solver_settings": solver_settings,
            "dynamics": dynamics,
            "grid": grid,
        }
        return problem_definition

    def get_solver_settings(self, accuracy="low", mode="brt"):
        accuracies = ["low", "medium", "high", "very_high"]
        modes = ["brs", "brt"]

        if accuracy not in accuracies:
            print(f"'accuracy' must be one of {[accuracies]}")

        if mode not in modes:
            print(f"'mode' must be one of {[modes]}")

        if mode == "brs":
            return hj.SolverSettings.with_accuracy(accuracy)
        elif mode == "brt":
            return hj.SolverSettings.with_accuracy(
                accuracy, hamiltonian_postprocessor=hj.solver.backwards_reachable_tube
            )

    def get_domain_grid(self, domain, domain_cells):
        """
        :param domain: [[-min values-], [-max values-]]
        :param domain_cells: [each domain dimension number of cells]
        """
        grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
            hj.sets.Box(
                np.array(domain[0, :]),
                np.array(domain[1, :]),
            ),
            tuple(domain_cells),
            periodic_dims=2,
        )
        return grid

    def compute_warm_start_values(self, grid_map: np.ndarray, dx) -> np.ndarray:
        """
        combines the previous solution V_last(x) to l(x) of the new map
        :param grid_map: 2D numpy array of occupancy values
        :return: system-dim-D numpy array of warm-started values. e.g. for dubins3d, this is a 3D array
        """
        warm_values = self.last_values
        l_x = self.compute_initial_values(grid_map, dx)
        changed = np.where(self.last_grid_map != grid_map)
        # TODO: Implement loading previous values from file
        try:
            warm_values = warm_values.at[changed].set(
                l_x[changed]
            )  # jax syntax for 'warm_values[changed] = l_x[changed]'
        except AttributeError as e:
            warm_values[changed] = l_x[changed]
        return warm_values

    def compute_initial_values(
        self, grid_map: np.ndarray, dx: float = 0.1
    ) -> np.ndarray:
        """
        computes l(x) where grid_map is an occupancy map assuming 0 = occupied, 1 = free
        :param grid_map: 2D numpy array of occupancy values
        :param dx: float, grid spacing

        :return: system-dim-D numpy array of initial values. e.g. for dubins3d, this is a 3D array
        """
        if self.config.system_name == "dubins3d":
            initial_values = (
                grid_map - 0.5
            )  # offset to make sure the distance is 0 at the border
            initial_values = skfmm.distance(initial_values, dx=dx)
            initial_values = np.tile(
                initial_values[:, :, np.newaxis], (1, 1, self.config.domain_cells[2])
            )
            return initial_values
        elif self.config.system_name == "dubins3d_velocity":
            initial_values = grid_map - 0.5
            initial_values = skfmm.distance(initial_values, dx=dx)
            initial_values = np.tile(
                initial_values[:, :, np.newaxis, np.newaxis],
                (1, 1, self.config.domain_cells[2], self.config.domain_cells[3]),
            )
            return initial_values
        else:
            raise NotImplementedError(
                f"System {self.config.system_name} not implemented"
            )

    def step(self, problem_definition, initial_values, time, target_time):
        target_values = hj.step(
            **problem_definition,
            time=time,
            values=initial_values,
            target_time=target_time,
            progress_bar=False,
        )
        return target_values

    def solve(
        self,
        grid_map,
        map_resolution,
        time=0.0,
        target_time=-10.0,
        dt=0.01,
        epsilon=0.0001,
    ):
        print("Grid map shape:", grid_map.shape) if self.config.print_progress else None
        (
            print("Domain cells:", self.config.domain_cells)
            if self.config.print_progress
            else None
        )

        if self.last_values is None:
            (
                print("Computing value function from scratch")
                if self.config.print_progress
                else None
            )
            initial_values = self.compute_initial_values(grid_map, map_resolution)
        else:
            (
                print("Computing warm-started value function")
                if self.config.print_progress
                else None
            )
            initial_values = self.compute_warm_start_values(grid_map, map_resolution)

        self.initial_values = initial_values

        self.last_grid_map = grid_map

        if self.problem_definition is None:
            self.problem_definition = self.get_problem_definition(
                self.config.system_name,
                self.config.domain,
                self.config.domain_cells,
                self.config.mode,
                self.config.accuracy,
            )

        times = np.linspace(time, target_time, int((-target_time - time) / dt))

        print("Starting BRT computation") if self.config.print_progress else None
        time_start = time_pkg()
        for i in range(1, len(times)):
            time = times[i - 1]
            target_time = times[i]
            values = self.step(
                self.problem_definition, initial_values, time, target_time
            )

            if self.config.until_convergent:
                diff = np.amax(np.abs(values - initial_values))
                print("diff: ", diff) if self.config.print_progress else None
                if diff < epsilon:
                    (
                        print("Converged fast, lucky you!")
                        if self.config.print_progress
                        else None
                    )
                    break

            initial_values = values

            if self.config.print_progress:
                print(f"Current times step: {time} s to {target_time} s")
                print(f"Max absolute difference between V_prev and V_now = {diff}")
        time_end = time_pkg()
        (
            print(f"Time taken: {time_end - time_start} seconds")
            if self.config.print_progress
            else None
        )

        self.last_values = values

        return np.array(values)

    def check_if_safe(self, state, values):
        """
        checks if `state` is safe given most recent Value function
        """
        # TODO: modify for other systems
        state_ind = self._state_to_grid(state)
        value = values[*state_ind]
        try:
            initial_value = self.initial_values[*state_ind]
        except TypeError as e:
            initial_value = None

        return value > self.config.superlevel_set_epsilon, value, initial_value

    def compute_safe_control(
        self, state, nominal_action, action_bounds, values=None, values_grad=None
    ):
        """
        Args:
            state (np.ndarray): [x, y, theta, v] state of the system
            nominal_action (np.ndarray): [angular_velocity, linear_acceleration] action
            action_bounds (np.ndarray): [[min_angular_velocity, max_angular_velocity], [min_linear_acceleration, max_linear_acceleration]]
            values (np.ndarray, optional): values from the level set, where > 0 is safe and < 0 is unsafe. Defaults to None.
            values_grad (np.ndarray, optional): gradient of the values. Defaults to None.
        """
        if values is None:
            values = self.last_values

        if values_grad is None:
            values_grad = np.gradient(values)

        if self.problem_definition is None:
            self.problem_definition = self.get_problem_definition(
                self.config.system_name,
                self.config.domain,
                values.shape,
                self.config.mode,
                self.config.accuracy,
            )

        state = np.array(state)
        is_safe, value, initial_value = self.check_if_safe(state, values)

        action = nominal_action.copy()
        has_intervened = not is_safe

        # If the current state is not safe, compute a safe action
        if not is_safe:
            # Get the state indices in the grid
            state_ind = self._state_to_grid(state)
            
            # Extract the gradient at the current state
            grad_x = values_grad[0][*state_ind]
            grad_y = values_grad[1][*state_ind]
            grad_theta = values_grad[2][*state_ind]
            grad_v = values_grad[3][*state_ind]
            
            # Gradient of the value function at the current state
            grad_V = np.array([grad_x, grad_y, grad_theta, grad_v])
            
            # Extract state components
            x, y, theta, v = state
            
            # Define the dynamics of the system
            # Open loop dynamics: [v*cos(theta), v*sin(theta), 0, 0]
            # Control jacobian: [[0, 0], [0, 0], [1, 0], [0, 1]]
            
            # Set up the optimization problem
            u = cp.Variable(2)  # Control input [angular_velocity, linear_acceleration]
            
            # Objective: Minimize the squared distance between u and nominal_action
            objective = cp.Minimize(cp.sum_squares(u - nominal_action))
            
            # Constraint: The control should move the system towards the safe set
            # This means the time derivative of V should be non-negative: dV/dt >= 0
            # dV/dt = grad_V^T * (f(x) + g(x)*u)
            # where f(x) is the open loop dynamics and g(x) is the control jacobian
            
            # Open loop dynamics contribution
            f_x = np.array([v * np.cos(theta), v * np.sin(theta), 0.0, 0.0])
            
            # Control jacobian
            g_x = np.array([
                [0, 0],
                [0, 0],
                [1, 0],
                [0, 1]
            ])
            
            # Safety constraint: grad_V^T * (f(x) + g(x)*u) >= 0
            # Simplify to: grad_V^T * f(x) + grad_V^T * g(x) * u >= 0
            # Further simplify to: grad_V^T * g(x) * u >= -grad_V^T * f(x)
            
            lhs = grad_V @ g_x  # Left-hand side coefficient for u
            rhs = -grad_V @ f_x  # Right-hand side constant term
            
            # Add safety constraint
            constraints = [lhs @ u >= rhs]
            
            # Add control bounds
            constraints.append(u[0] >= action_bounds[0, 0])  # min angular velocity
            constraints.append(u[0] <= action_bounds[0, 1])  # max angular velocity
            constraints.append(u[1] >= action_bounds[1, 0])  # min linear acceleration
            constraints.append(u[1] <= action_bounds[1, 1])  # max linear acceleration
            
            # Solve the optimization problem
            prob = cp.Problem(objective, constraints)
            try:
                prob.solve()
                
                # Check if the problem was solved successfully
                if prob.status == cp.OPTIMAL or prob.status == cp.OPTIMAL_INACCURATE:
                    # Update the action with the safe control
                    action = u.value
                else:
                    # If optimization failed, use a simple backup strategy:
                    # Move in the direction of the gradient of the value function
                    # This is a simple approach that may not be optimal but should improve safety
                    control_direction = lhs / (np.linalg.norm(lhs) + 1e-6)  # Normalize
                    
                    # Scale the control direction to fit within bounds
                    action[0] = np.clip(nominal_action[0] + 0.5 * control_direction[0], 
                                       action_bounds[0, 0], action_bounds[0, 1])
                    action[1] = np.clip(nominal_action[1] + 0.5 * control_direction[1], 
                                       action_bounds[1, 0], action_bounds[1, 1])
            except Exception as e:
                print(f"Optimization failed: {e}")
                # Fall back to nominal action if optimization fails completely
                action = nominal_action

        return action, value, initial_value, has_intervened

    def _state_to_grid(self, state):
        grid = self.problem_definition["grid"]
        state_ind = np.array(grid.nearest_index(state)) - 1
        return state_ind

    def plot_zero_level(
        self,
        grid_data,
        grid_map=None,
        fig=None,
        ax=None,
        title="Contour Map with 0-Level Set",
    ):
        if fig is None or ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        if grid_map is not None:
            x = np.linspace(
                self.config.domain[0][0],
                self.config.domain[1][0],
                self.config.domain_cells[0],
            )
            y = np.linspace(
                self.config.domain[0][1],
                self.config.domain[1][1],
                self.config.domain_cells[1],
            )

            # Create the meshgrid
            x, y = np.meshgrid(x, y)

            # Plot the data
            plt.pcolormesh(x, y, grid_map, shading="auto", cmap="gray")

        x = np.linspace(
            self.config.domain[0][0],
            self.config.domain[1][0],
            self.config.domain_cells[0],
        )
        y = np.linspace(
            self.config.domain[0][1],
            self.config.domain[1][1],
            self.config.domain_cells[1],
        )

        x, y = np.meshgrid(x, y)
        z = grid_data

        # plot all contours
        contour_all = plt.contour(x, y, z, levels=20, cmap="viridis")
        plt.clabel(contour_all, inline=True, fontsize=8)

        # plot the 0-level set in red
        contour = plt.contour(x, y, z, levels=[0], colors="red")
        plt.clabel(contour, fmt="%2.1f", colors="black")

        plt.title(title)
        plt.xlabel("X axis")
        plt.ylabel("Y axis")
        plt.xticks([])
        plt.yticks([])
        plt.show()
        # plt.gca().set_aspect('equal')

        plt.close(fig)
