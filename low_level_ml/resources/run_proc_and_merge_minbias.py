#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py3-v4.1.1/icetray-start
#METAPROJECT combo/stable
from glob import glob
import os
import subprocess
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-i", nargs="+", dest="infiles", required=True)
parser.add_argument("-o", dest="outdir", required=True)
args = parser.parse_args()

script_path = "/data/user/chaack/software/low-level-ml/low_level_ml/resources/proc_and_merge_minbias.py"
def proc_file(f):
    outfile = os.path.join(
        args.outdir,
        os.path.basename(f))
    subprocess.run(args=" ".join(["/cvmfs/icecube.opensciencegrid.org/py3-v4.1.1/icetray-env",
                         "combo/stable",
                         "python",
                         script_path,
                         f"-i {f}", f"-o {outfile}"]),
                   check=True,
                   shell=True,
                   env={}
                  )

for f in args.infiles:
    proc_file(f)
