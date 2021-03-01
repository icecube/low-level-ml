#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py3-v4.1.1/icetray-start
#METAPROJECT combo/stable
"""
Processing script to select minbias events from MC
"""
from argparse import ArgumentParser
from icecube import dataio, icetray
from I3Tray import *

def filter_minbias(fr):
    return ("FilterMask_NullSplit0" in fr) and (fr["FilterMask_NullSplit0"]["FilterMinBias_13"].prescale_passed)

parser = ArgumentParser()
parser.add_argument("-i", required=True, nargs="+", dest="infiles")
parser.add_argument("-o", required=True, dest="outfile")
args = parser.parse_args()

tray = I3Tray()
tray.Add("I3Reader", FilenameList=args.infiles)
tray.Add(filter_minbias, Streams=[icetray.I3Frame.Physics])
tray.Add(
    "I3Writer",
    Filename=args.outfile,
    DropOrphanStreams=[icetray.I3Frame.DAQ])
tray.Execute()

# Write a summary file

with open(args.outfile+".summary", "w") as summ:
    summ.writelines([line+"\n" for line in args.infiles])

