import re
import socket
import subprocess
from absl import flags

# FLAGS = flags.FLAGS
# flags.DEFINE_float("timeout_secs", 60, "Seconds to wait for bot to respond")

class WBridge5Client(object):
  """Manages the connection to a WBridge5 bot."""

  def __init__(self, command):
    self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self.sock.bind(("", 0))
    self.port = self.sock.getsockname()[1]
    self.sock.listen(1)
    self.process = None
    self.command = command.format(port=self.port)

  def start(self):
    if self.process is not None:
      self.process.kill()
    self.process = subprocess.Popen(self.command.split(" "))
    self.conn, self.addr = self.sock.accept()

  def read_line(self):
    line = ""
    while True:
      self.conn.settimeout(60)
      data = self.conn.recv(1024)
      if not data:
        raise EOFError("Connection closed")
      line += data.decode("ascii")
      if line.endswith("\n"):
        return re.sub(r"\s+", " ", line).strip()

  def send_line(self, line):
    # print("send: ", line)
    self.conn.send((line + "\r\n").encode("ascii"))