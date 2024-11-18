"""
This file defines a StylePacket class, which stores shared information used in
multiple plots. The idea is that, when generating several plots for a document, 
you might want to consistently use a particular font, background color, etc., 
and it would be inconvenient to specify all of those things each time. The 
StylePacket object conveniently stores that kind of information and provides
methods that wrap various Matplotlib functions.
"""

from dataclasses import dataclass


@dataclass
class StylePacket:
    pass
