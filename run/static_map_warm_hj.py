from learning.base_model import BaseModel
from learning.gaussian_process import GaussianProcess
from itertools import product
import matplotlib.pyplot as plt
from src.failure_map_builder import FailureMapBuilder, FailureMapParams
from envs.smoke_env import EnvParams, RobotParams, SmokeEnv
from reachability.warm_start_solver import WarmStartSolver, WarmStartSolverConfig
from simulator.static_smoke import SmokeBlobParams
import numpy as np

from src.mppi import Navigator, dubins_dynamics_tensor

def main():
    env_params = EnvParams()
    env_params.world_x_size = 120
    env_params.world_y_size = 100
    env_params.max_steps = 2000
    env_params.render = True
    env_params.goal_location = (50, 50)

    robot_params = RobotParams()
    smoke_blob_params = [
        SmokeBlobParams(x_pos=10, y_pos=80, intensity=1.0, spread_rate=15.0),
        SmokeBlobParams(x_pos=80, y_pos=10, intensity=1.0, spread_rate=15.0),
        SmokeBlobParams(x_pos=20, y_pos=20, intensity=1.0, spread_rate=6.0),
        SmokeBlobParams(x_pos=50, y_pos=50, intensity=1.0, spread_rate=5.0),
        SmokeBlobParams(x_pos=70, y_pos=70, intensity=1.0, spread_rate=8.0)
    ]

    env = SmokeEnv(env_params, robot_params, smoke_blob_params)
    state, _ = env.reset(initial_state=np.array([10.0, 10.0, 0.0, 0.0]))

    learner = GaussianProcess()
    
    builder = FailureMapBuilder(
        params=FailureMapParams(
            x_size=env_params.world_x_size, 
            y_size=env_params.world_y_size, 
            resolution=1.0, 
            map_rule_type='threshold',
            map_rule_threshold=0.7
            )
        )

    solver = WarmStartSolver(
        config=WarmStartSolverConfig(
            system_name="dubins3d",
            domain_cells=[120, 100, 100],
            domain=[[0, 0, 0], [env_params.world_x_size, env_params.world_y_size, 2*np.pi]],
            mode="brt",
            accuracy="low",
            converged_values=None,
            until_convergent=True,
            print_progress=False,
        )
    )

    navigator = Navigator()
    navigator.set_odom(state[:2],state[2])
    navigator.set_map(builder.failure_map, [120, 100], [0, 0], 1.0)
    navigator.set_goal(list(env_params.goal_location))

    nominal_action_w = navigator.get_command().item()
    nominal_action = np.array([2, nominal_action_w])

    update_interval = 10
    values = None

    # Real time plotting
    f = plt.figure()
    map_ax = f.add_subplot(111)

    x = np.linspace(solver.config.domain[0][0], solver.config.domain[1][0], solver.config.domain_cells[0])
    y = np.linspace(solver.config.domain[0][1], solver.config.domain[1][1], solver.config.domain_cells[1])

    x, y = np.meshgrid(x, y)

    map_cax = map_ax.imshow(builder.failure_map, extent=[0, env_params.world_x_size, 0, env_params.world_y_size], origin='lower', cmap='gray', vmin=0, vmax=1)
    bar = f.colorbar(map_cax, ax=map_ax, label="Failure")
    map_ax.set_title("Failure Map")
    plt.draw()

    f_continuous = plt.figure()
    ax_continuous = f_continuous.add_subplot(111)
    continuous_map_ax = ax_continuous.imshow(builder.build_continuous_map(learner), extent=[0, env_params.world_x_size, 0, env_params.world_y_size], origin='lower', cmap='gray', vmin=0, vmax=1)
    ax_continuous.set_title("Continuous Map")
    bar_continuous = f_continuous.colorbar(continuous_map_ax, ax=ax_continuous, label="Failure")
    plt.draw()

    # Create a new figure for the gradient plot
    f_gradient = plt.figure()
    x = np.arange(env_params.world_x_size)
    y = np.arange(env_params.world_y_size)
    X, Y = np.meshgrid(x, y)
    ax = f_gradient.add_subplot(111)
    gradient_ax = ax.quiver(X, Y, np.zeros(X.shape), np.zeros(Y.shape), scale=50)
    ax.set_title("Gradient")
    plt.draw()

    # Create a line plot for the gradient values

    for t in range(1,env_params.max_steps):
        learner.track_data(state[0:2][::-1], state[3])

        if t % update_interval == 0:
            learner.update()
            failure_map = builder.build_map(learner)
            continuous_map = builder.build_continuous_map(learner)
            map_cax.set_array(builder.failure_map)

            continuous_map_ax.set_array(continuous_map)
            plt.draw()
            if np.all(failure_map == 1):
                values = None
            else:
                values = solver.solve(failure_map.T, target_time=-1.0, dt=0.1, epsilon=0.0001)

        nominal_action = navigator.get_command()
        nominal_action = np.array([2.0, nominal_action.item()])

        if values is not None:
            grad_values = np.gradient(values)

            gx, gy = grad_values[0][:,:,-1], grad_values[1][:,:,-1]

            for coll in map_ax.collections:
                coll.remove()

            state_ind = solver._state_to_grid(state[:3])
            z = values[:,:,state_ind[2]].T
            contour = map_ax.contour(x, y, z, levels=[0], colors="red")
            map_ax.clabel(contour, fmt="%2.1f", colors="black")

            gradient_ax.set_UVC(gx.T, gy.T)
            
            plt.draw()

            safe_action, _, _ = solver.compute_safe_control(state[0:3], nominal_action, action_bounds=np.array([[0.0, 5.0], [-4.0, 4.0]]), values=values)
        else:   
            safe_action = nominal_action

        state, reward, terminated, truncated, info = env.step(safe_action)
        navigator.set_odom(state[:2], state[2])

    env.close()


if __name__ == "__main__":
    main()