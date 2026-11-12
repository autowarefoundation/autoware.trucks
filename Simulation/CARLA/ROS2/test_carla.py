import carla

def print_spawn_points():
    client = carla.Client('localhost', 2000)
    world = client.load_world('Town04')
    spawn_points = world.get_map().get_spawn_points()
    print('Number of spawn points:', len(spawn_points))

def print_map_list():
    # Connect to the local CARLA server
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)

    # Get the list of all available map assets on this server
    available_maps = client.get_available_maps()

    print("Available Maps / Towns:")
    for map_path in sorted(available_maps):
        print(map_path)

def test_vehicle():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    bp = world.get_blueprint_library().find('vehicle.lincoln.mkz_2020')
    spawn_points = world.get_map().get_spawn_points()
    vehicle = world.try_spawn_actor(bp, spawn_points[0])

    physics = vehicle.get_physics_control()
    for i, wheel in enumerate(physics.wheels):
        print(f"Wheel {i}: max_steer_angle = {wheel.max_steer_angle} degrees")

    vehicle.destroy()

if __name__ == "__main__":
    # print_map_list()
    # print_spawn_points()
    test_vehicle()