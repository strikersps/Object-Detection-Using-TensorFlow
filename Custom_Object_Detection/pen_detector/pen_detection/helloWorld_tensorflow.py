import tensorflow as tf
from packaging import version
import os

if version.parse(tf.__version__) < version.parse("1.4.0"):
    raise ImportError("Install the tensorflow version > 1.4.0")
else:
    print("Tensorflow Version: %s" % (tf.__version__))

print(os.system('pip3 install --upgrade tensorflow'))