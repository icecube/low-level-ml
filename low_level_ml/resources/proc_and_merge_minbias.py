#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py3-v4.1.1/icetray-start
#METAPROJECT combo/stable
"""
!!!ONLY USE WITH CORSIKA SET 20904!!!
Select minbias events and merge the SignalI3MCTree from the detector level files
"""
import os
print(os.environ)
from argparse import ArgumentParser
from icecube import dataio, icetray, phys_services
from I3Tray import *

def filter_minbias(fr):
    return ("FilterMask_NullSplit0" in fr) and (fr["FilterMask_NullSplit0"]["FilterMinBias_13"].prescale_passed)


class MergeMCTree(icetray.I3Module):
    def __init__(self, context):
        super().__init__(context)
        self.AddParameter("filename", "", None)

    def Configure(self):
        self._filename = self.GetParameter("filename")
        self._hdl = dataio.I3File(self._filename)
        self.not_found_cnt = 0

    def DAQ(self, fr):
        penergy = fr["I3MCTree_preMuonProp"].get_primaries()[0].energy

        found = False

        for fr2 in self._hdl:
            if fr2.Stop != icetray.I3Frame.DAQ:
                continue
            penergy2 = fr2["I3MCTree_preMuonProp"].get_primaries()[0].energy
            if penergy == penergy2:
                fr["SignalI3MCTree"] = fr2["SignalI3MCTree"]
                found = True
                break
        if not found:
            self._hdl.rewind()
            self.not_found_cnt += 1
        self.PushFrame(fr)

    def Finish(self):
        print(f"Not found {self.not_found_cnt}")


parser = ArgumentParser()
parser.add_argument("-i", required=True, dest="infile")
parser.add_argument("-o", required=True, dest="outfile")
args = parser.parse_args()

genfile = args.infile.replace(
    "/data/sim/IceCube/2016/filtered/level2/",
    "/data/sim/IceCube/2020/generated/")
genfile = genfile.replace("Level2_IC86.2016", "/detector/IC86.2020")


tray = I3Tray()
tray.Add("I3Reader", Filename=args.infile)
tray.Add(filter_minbias, Streams=[icetray.I3Frame.Physics])
tray.Add("I3OrphanQDropper")
tray.Add(MergeMCTree, filename=genfile)


tray.Add(
    "I3Writer",
    Filename=args.outfile,
    DropOrphanStreams=[icetray.I3Frame.DAQ])

tray.Execute()
