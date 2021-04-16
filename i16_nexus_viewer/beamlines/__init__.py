"""
Beamlines
"""

from .i16 import i16

_beamline_list = [i16]

beamlines = {bm.name: bm for bm in _beamline_list}
