import glob
import sys
import os
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import numpy as np
import random
import time

import argparse
from datetime import datetime
from IPython.display import display, clear_output


import carla
from carla import ColorConverter as cc
from manual_control import World, HUD

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_b
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_g
    from pygame.locals import K_h
    from pygame.locals import K_i
    from pygame.locals import K_l
    from pygame.locals import K_m
    from pygame.locals import K_n
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_v
    from pygame.locals import K_w
    from pygame.locals import K_x
    from pygame.locals import K_z
    from pygame.locals import K_MINUS
    from pygame.locals import K_EQUALS
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')


# def make_folder():
#     # We create a dataset stored in folder name with current date and time
    
    
#     return directory

def spawn_vehicle(world):
    blueprint_car = world.get_blueprint_library().find('vehicle.tesla.model3')
    blueprint_car.set_attribute('role_name', 'hp')
    color = random.choice(blueprint_car.get_attribute('color').recommended_values)

    blueprint_car.set_attribute('color', color)

    spawn_points = world.get_map().get_spawn_points()
    number_of_spawn_points = len(spawn_points)

    # Random shuffle spawn points. But if need to start on a particular point every time, then comment line below
    if 0 < number_of_spawn_points:
        random.shuffle(spawn_points)
        transform = spawn_points[0]
        #spawn our vehicle !
        vehicle = world.spawn_actor(blueprint_car,transform)
        print('\nVehicle spawned')
    else: 
        #no spawn points 
        logging.warning('Could not found any spawn points')
    return vehicle

# Reference: https://carla.readthedocs.io/en/latest/core_sensors/
def camera_sensor(world, vehicle, WIDTH, HEIGHT):
    camera_blueprint_car = None
    camera_blueprint_car = world.get_blueprint_library().find("sensor.camera.rgb")
    
    camera_blueprint_car.set_attribute("image_size_x", str(WIDTH))
    camera_blueprint_car.set_attribute("image_size_y", str(HEIGHT))
    camera_blueprint_car.set_attribute("fov", str(105))

    camera_location = carla.Location(2,0,1)
    camera_rotation = carla.Rotation(0,0,0)
    
    camera = world.spawn_actor(camera_blueprint_car, carla.Transform(camera_location, camera_rotation), 
                               attach_to = vehicle, attachment_type = carla.AttachmentType.Rigid)
    
    return camera

def image_processing(image):
    image_bgra = image.raw_data
    # Image obtained is 32-bit per pixel. Need to compress it 8-bit
    image_bgra_8bit = np.array(image_bgra, dtype = np.dtype("uint8"))
    image_bgra_8bit = np.reshape(image_bgra_8bit, (image.height, image.width, 4))
    # reshape image to image width, height.
    print(image_bgra_8bit)

    process_image = image_bgra_8bit[:,:,:3]/255

def save_image(vehicle, directory, image_file, controls_file, image_from_carla):
    image = image_processing(image_from_carla)
    vehicle_controls = vehicle.get_control()
    control_data = [vehicle_controls.steer, vehicle_controls.throttle, vehicle_controls.brake]    
    np.save(image_file, image)
    np.save(controls_file, control_data)
    
def game_loop(args):

    pygame.init()
    pygame.font.init()
    world = None

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)

    display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
    display.fill((0,0,0))
    pygame.display.flip()

    hud = HUD(args.width, args.height)
    world = World(client.get_world(), hud, args)

    today = datetime.now()
    h = "0"+str(today.hour) if today.hour < 10 else str(today.hour)
    m = "0"+str(today.minute) if today.minute < 10 else str(today.minute)

    # Directory name representing dataset creation in format YYYYMMDD_HHMM
    directory = "TEST_DATA/" + today.strftime('%Y%m%d_')+ h + m + "_npy"
    print(directory)

    if not os.path.exists(directory):
        os.makedirs(directory)

    try:
        image_file = open(directory + "/images.npy","ba+") 
        controls_file = open(directory + "/controls.npy","ba+")     
    except:
        print("Files could not be opened")
    
    vehicle = spawn_vehicle(world.world)
    camera = camera_sensor(world.world, vehicle, WIDTH=640, HEIGHT=480)

    vehicle.set_autopilot(True)

    camera.listen(lambda data: save_image(vehicle, directory, image_file, controls_file, data))

    # try:
    i = 0
    clock = pygame.time.Clock()
    #How much frames do we want to save
    while i < 25000:
        
        clock.tick_busy_loop(60)
        # if controller.parse_events(client, world, clock):
        #     return
        world.tick(clock)
        world.render(display)
        pygame.display.flip()


        # world_snapshot = world.world.wait_for_tick()
        # clear_output(wait=True)
        # display(f"{str(i)} frames saved")
        i += 1
    # except:
    #     print('\nSimulation error.')
    
    #Destroy everything     
    if vehicle is not None:
        if camera is not None:
            camera.stop()
            camera.destroy()
    vehicle.destroy()

    #Close everything   
    image_file.close()
    controls_file.close()
    print("Data retrieval finished")
    print(directory)

def main():
    argparser = argparse.ArgumentParser(description='Control Agent')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--rolename',
        metavar='NAME',
        default='hero',
        help='actor role name (default: "hero")')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    game_loop(args) 

if __name__ == '__main__':

    main()
    