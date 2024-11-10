import os
import sys
import importlib
sys.path.insert(0, os.path.join(os.path.split(os.path.abspath(__file__))[0], "plotomata"))

import plotters
import colors

importlib.reload(plotters)
importlib.reload(colors)

from plotters import *
from colors import *
