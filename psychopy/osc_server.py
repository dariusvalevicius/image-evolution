"""Small example OSC server

This program listens to several addresses, and prints some information about
received packets.
"""
import argparse
import math

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
  with open("eda_data.txt",'a') as f:
        now = datetime.now().time().strftime('%H:%M:%S.%f')[:-3]
        string = f"{args[0]},{now}\n"
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


  ## Reset data file
  with open("eda_data.txt", 'w') as file:
    pass

  dispatcher = Dispatcher()
#   dispatcher.map("/filter", print)
#   dispatcher.map("/*", print)
  dispatcher.map("/EmotiBit/0/EDA", print)
  dispatcher.map("/EmotiBit/0/EDA", append_to_file)
  # dispatcher.map("/EmotiBit/0/SCR:AMP", print)
  # dispatcher.map("/EmotiBit/0/SCR:AMP", append_to_file)
  dispatcher.map("/volume", print_volume_handler, "Volume")
  dispatcher.map("/logvolume", print_compute_handler, "Log volume", math.log)

  server = osc_server.ThreadingOSCUDPServer(
      (args.ip, args.port), dispatcher)
  print("Serving on {}".format(server.server_address))
  server.serve_forever()