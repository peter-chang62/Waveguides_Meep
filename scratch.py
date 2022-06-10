import meep as mp
import meep.materials as mt
import numpy as np
import clipboard_and_style_sheet
from meep import mpb
import matplotlib.pyplot as plt
import utilities as util
import h5py
import time

clipboard_and_style_sheet.style_sheet()


class RidgeWaveGuide:
    def __init__(self, width, height, substrate_medium, waveguide_medium):
        # passed as arguments from __init__()
        self.wdth = width
        self.hght = height
        self.sbstrt_mdm = substrate_medium
        self.wvgd_mdm = waveguide_medium

        # additional attributes
        self._sy = 8
        self._sz = 8

        self._hght_sbstrt = (self.sy / 2) - (self.hght / 2)

        self.blk_wvgd = mp.Block(size=mp.Vector3(mp.inf, self.wdth, self.hght))
