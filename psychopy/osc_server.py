"""Small example OSC server

This program listens to several addresses, and prints some information about
received packets.
"""
import argparse
import math
import os

from pythonosc.dispatcher import Dispatcher
from pythonosc import osc_server
from datetime import datetime

def print_volume_handler(unused_addr, args, volume):
  print("[{0}] ~ {1}".format(args[0], volume))

def print_compute_handler(unused_addr, args, volume):
  try:
    print("[{0}] ~ {1}".format(args[0], args[1](volume)))
  except ValueError: pass

def append_to_file(unused_addr, *args):

  if not os.path.exists('stream'):
    os.makedirs('stream')

  with open(f'stream/{args[0][0]}','a') as f:
        now = datetime.now().time().strftime('%H:%M:%S.%f')[:-3]
        string = f"{args[1]},{now}\n"
        f.write(string)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--ip",
      default="127.0.0.1", help="The ip to listen on")
  parser.add_argument("--port",
      type=int, default=12345, help="The port to listen on")
  # parser.add_argument("--file",
  #     type=str, default="psychopy/eda_data.txt", help="The port to listen on")
  args = parser.parse_args()


  # ## Reset data file
  # with open("eda_data.txt", 'w') as file:
  #   pass

  dispatcher = Dispatcher()
#   dispatcher.map("/filter", print)
#   dispatcher.map("/*", print)
  # dispatcher.map("/EmotiBit/0/EDA", print)
  dispatcher.map("/EmotiBit/0/EDA", append_to_file, "eda_data.txt")
  # dispatcher.map("/EmotiBit/0/SCR:AMP", print)
  dispatcher.map("/EmotiBit/0/SCR:AMP", append_to_file, "samp_data.txt")
  # dispatcher.map("/EmotiBit/0/TEMP", print)
  dispatcher.map("/EmotiBit/0/TEMP", append_to_file, "temp_data.txt")
  # dispatcher.map("/EmotiBit/0/HR", print)
  dispatcher.map("/EmotiBit/0/HR", append_to_file, "hr_data.txt")


  dispatcher.map("/volume", print_volume_handler, "Volume")
  dispatcher.map("/logvolume", print_compute_handler, "Log volume", math.log)

  server = osc_server.ThreadingOSCUDPServer(
      (args.ip, args.port), dispatcher)
  print("Serving on {}".format(server.server_address))
  server.serve_forever()