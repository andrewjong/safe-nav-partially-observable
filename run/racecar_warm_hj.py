import numpy as np

import posggym
from reachability.warm_start_solver import (WarmStartSolver,
                                            WarmStartSolverConfig)
from src.mppi import Navigator, dubins_dynamics_tensor


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

    observations, infos = env.reset()


    # creates a failure map with the given width, height and resolution.
    # unobserved cells are initialized as fail set.

    failure_map = FailureMapBuilder(MAP_WIDTH, MAP_HEIGHT, MAP_RESOLUTION)
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
        failure_map.update_from_lidar(lidar_distances, vehicle_x, vehicle_y, vehicle_angle)



        if all_done:
            observations, infos = env.reset()

    env.close()


if __name__ == "__main__":
    main()