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

  with open(args[0][0],'a') as f:
        now = datetime.now().time().strftime('%H:%M:%S.%f')
        string = f"{args[1]}\t{now}\n"
        f.write(string)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--output", help="Streaming data output path")
  parser.add_argument("--ip",
      default="127.0.0.1", help="The ip to listen on")
  parser.add_argument("--port",
      type=int, default=12345, help="The port to listen on")
  args = parser.parse_args()

  dispatcher = Dispatcher()
#   dispatcher.map("/filter", print)
#   dispatcher.map("/*", print)
  # dispatcher.map("/EmotiBit/0/EDA", print)
  # dispatcher.map("/EmotiBit/0/EDA", append_to_file, "eda_data.txt")
  dispatcher.map("/EmotiBit/0/SCR:AMP", print)
  dispatcher.map("/EmotiBit/0/SCR:AMP", append_to_file, os.path.join(args.output, "scr_streaming_data.txt"))
  dispatcher.map("/EmotiBit/0/HR", print)
  dispatcher.map("/EmotiBit/0/HR", append_to_file, os.path.join(args.output, "hr_streaming_data.txt"))
  # dispatcher.map("/EmotiBit/0/TEMP", print)
  # dispatcher.map("/EmotiBit/0/TEMP", append_to_file, "temp_data.txt")
  # dispatcher.map("/EmotiBit/0/HR", print)
  # dispatcher.map("/EmotiBit/0/HR", append_to_file, "hr_data.txt")


  dispatcher.map("/volume", print_volume_handler, "Volume")
  dispatcher.map("/logvolume", print_compute_handler, "Log volume", math.log)


  server = osc_server.ThreadingOSCUDPServer(
      (args.ip, args.port), dispatcher)
  print("Serving on {}".format(server.server_address))
  server.serve_forever()