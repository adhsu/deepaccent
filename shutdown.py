# shutdown.py
from subprocess import call

def shutdown():
  call(['sudo','bash','./shutdown-script.sh'])