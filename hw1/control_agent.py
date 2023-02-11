import sys
import os
sys.path.append(os.getcwd())

import argparse
import datetime
import carla

def make_folder():
    # We create a dataset stored in folder name with current date and time
    today = datetime.now()
    h = "0"+str(today.hour) if today.hour < 10 else str(today.hour)
    m = "0"+str(today.minute) if today.minute < 10 else str(today.minute)

    # Directory name representing dataset creation in format YYYYMMDD_HHMM
    directory = "TEST_DATA/" + today.strftime('%Y%m%d_')+ h + m + "_npy"
    print(directory)

    if not os.path.exists(directory):
        os.makedirs(directory)

def spawn_vehicle(world):
    blueprint_car = world.get_blueprint_library().filter('vehicle.tesla.model3')
    blueprint_car.set_attribute('color', "black")
    

def game_loop(args):
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    client.reload_world()

    vehicle = None
    camera = None
    world = client.get_world()



def main():
    argparser = argparse.ArgumentParser(description='Control Agent')
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
    args = argparser.parse_args()

    game_loop(args) 
    make_folder()  
    