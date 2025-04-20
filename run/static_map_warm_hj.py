from learning.base_model import BaseModel
from learning.gaussian_process import GaussianProcess
from itertools import product
import matplotlib.pyplot as plt
from src.failure_map_builder import GPFailureMapBuilder, FailureMapParams
from envs.smoke_env import EnvParams, RobotParams, SmokeEnv
from reachability.warm_start_solver import WarmStartSolver, WarmStartSolverConfig
from simulator.static_smoke import SmokeBlobParams
import numpy as np

from src.mppi import Navigator, dubins_dynamics_tensor
from matplotlib.patches import FancyArrow, Arrow

def main():
    env_params = EnvParams()
    env_params.world_x_size = 30
    env_params.world_y_size = 40
    env_params.max_steps = 2000
    env_params.render = False
    env_params.goal_location = (25, 37)

    robot_params = RobotParams()
    smoke_blob_params = [
        SmokeBlobParams(x_pos=15, y_pos=20, intensity=1.0, spread_rate=8.0),
    ]

    env = SmokeEnv(env_params, robot_params, smoke_blob_params)

    state, _ = env.reset(initial_state=np.array([5.0, 5.0, 0.0, 0.0]))

    learner = GaussianProcess()
    
    builder = GPFailureMapBuilder(
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
            domain_cells=[30, 40, 40],
            domain=[[0, 0, 0], [env_params.world_x_size, env_params.world_y_size, 2*np.pi]],
            mode="brt",
            accuracy="medium",
            converged_values=None,
            until_convergent=False,
            print_progress=False,
        )
    )

    nom_controller = Navigator()
    nom_controller.set_odom(state[:2],state[2])
    nom_controller.set_map(builder.failure_map, [30, 40], [0, 0], 1.0)
    nom_controller.set_goal(list(env_params.goal_location))

    # TODO: Make this part of the dynamics
    NOMINAL_ACTION_V = 2.0
    nominal_action_w = nom_controller.get_command().item()
    nominal_action = np.array([NOMINAL_ACTION_V, nominal_action_w])

    update_interval = 5
    values = None

    x = np.arange(env_params.world_x_size)
    y = np.arange(env_params.world_y_size)
    X, Y = np.meshgrid(x, y)

    f = plt.figure(figsize=(6, 6))
    gs = f.add_gridspec(1, 3)
    ax_env = f.add_subplot(gs[0])
    ax_map = f.add_subplot(gs[1])
    ax_fail = f.add_subplot(gs[2])

    learner_gt = GaussianProcess()

    for i in range(1000):
        X_sample = np.concatenate([np.random.uniform(0, env_params.world_y_size, 1), np.random.uniform(0, env_params.world_x_size, 1)])
        y_observe = env.smoke_simulator.get_smoke_density(X_sample[1], X_sample[0])
        learner_gt.track_data(X_sample, y_observe)
    learner_gt.update()


    x = np.linspace(0, env_params.world_x_size, env_params.world_x_size)
    y = np.linspace(0, env_params.world_y_size, env_params.world_y_size)
    xy = np.array(list(product(y, x)))

    y_pred, std = learner_gt.predict(xy)
    y_pred = y_pred.reshape(env_params.world_y_size, env_params.world_x_size)
    continuous_map = y_pred.reshape(int(env_params.world_y_size), int(env_params.world_x_size))
    gt_failure_map = builder.rule_based_map(continuous_map)

    plt.tight_layout()
    plt.draw()

    traj_nonfail = []
    traj_fail = []

    # Create a line plot for the gradient values
    for t in range(1,env_params.max_steps):
        learner.track_data(state[0:2][::-1], state[3])

        if t % update_interval == 0:
            learner.update()
            builder.build_map(learner)

            if np.all(builder.failure_map == 1):
                values = None
            else:
                values = solver.solve(builder.failure_map.T, target_time=-10.0, dt=0.1, epsilon=0.0001)

        nominal_action = nom_controller.get_command()
        nominal_action = np.array([NOMINAL_ACTION_V, nominal_action.item()])
        safe_action = nominal_action

        if values is not None:
            safe_action, _, _ = solver.compute_safe_control(state[0:3], nominal_action, action_bounds=np.array([[0.0, 5.0], [-4.0, 4.0]]), values=values)
        else:   
            safe_action = nominal_action

        state, reward, terminated, truncated, info = env.step(safe_action)

        if state[3] > builder.params.map_rule_threshold:
            traj_fail.append(state[0:2])
        else:
            traj_nonfail.append(state[0:2])

        if terminated:
            break

        nom_controller.set_odom(state[:2], state[2])
        # nom_controller.set_map(builder.failure_map, [30, 40], [0, 0], 1.0)

        # Real time plotting
        env._render_frame(fig=f, ax=ax_env)
        builder.plot_failure_map(fig=f, ax=ax_fail)

        learner.plot_map(x_size=env_params.world_x_size, y_size=env_params.world_y_size, fig=f, ax=ax_map)

        for arrow in ax_fail.patches:
            if isinstance(arrow, (FancyArrow, Arrow)):  
                arrow.remove()

        for coll in ax_fail.collections:
            coll.remove()

        if values is not None:
            is_safe, _, _ = solver.check_if_safe(state[:3], values)
            color_robot = 'g' if is_safe else 'r'
        else:
            color_robot = 'g'

        # Plot the agent's location as a blue arrow
        ax_fail.arrow(state[0], state[1], np.cos(state[2])*0.1, np.sin(state[2])*0.1, 
                                    head_width=1., head_length=1., fc=color_robot, ec=color_robot)
        
        if len(traj_nonfail) > 0:
            ax_fail.scatter(np.array(traj_nonfail)[:, 0], np.array(traj_nonfail)[:, 1], color='green', label='Non-fail', marker='.', s=0.2)
        if len(traj_fail) > 0:
            ax_fail.scatter(np.array(traj_fail)[:, 0], np.array(traj_fail)[:, 1], color='red', label='Fail', marker='.', s=0.2)

        if values is None:
            continue

        state_ind = solver._state_to_grid(state[:3])
        z = values[:,:,state_ind[2]].T
        z_mask = z > 0.1

        # contour = ax_fail.contour(x, y, z, levels=10, cmap='viridis')
        # ax_fail.clabel(contour, fmt="%2.1f", colors="black", fontsize=5)

        ax_fail.contour(x, y, z_mask, levels=[0.5], colors='green')

        ax_fail.contour(x, y, gt_failure_map, levels=[0.5], colors='red')

        plt.tight_layout()
        plt.draw()

    figure_name = input('Enter the name of the figure: ')
    f.savefig(f'misc/{figure_name}.png', bbox_inches='tight')

    print(f'Terminated at time {t*env_params.clock} seconds')
    print(f'Time for fail trajectory: {len(traj_fail) * env_params.clock} seconds')
    print(f'Time for non-fail trajectory: {len(traj_nonfail) * env_params.clock} seconds')

    env.close()


if __name__ == "__main__":
    main()