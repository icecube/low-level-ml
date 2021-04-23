import htcondor
import htcondor.dags
import os
import argparse
import sys
from glob import glob
from itertools import zip_longest
sys.path.append("../")
from datasets import datasets

parser = argparse.ArgumentParser()
parser.add_argument("-o", required=True, dest="outdir", type=str)
parser.add_argument("--dagname", required=True, dest="dagname", type=str)
parser.add_argument("--chunk_len", default=100, dest="chunk_len", type=int)
args=parser.parse_args()

script_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                           "../run_proc_and_merge_minbias.py")
script_path = script_path.replace("/mnt/lfs7/", "/data/")
submit_description=htcondor.Submit(
        {
            "executable": script_path,
            "arguments": "-i $(infiles) -o $(outdir)",
            "output": "logs/$(job).out",
            "error": "logs/$(job).err",
            "log": "logs/$(job).log",
            "when_to_transfer_output": "ON_EXIT",
            "should_transfer_files": "YES",
            "getenv": "False",
            "request_memory": "2GB"
        }
)
dag = htcondor.dags.DAG()

dataset = "CORSIKA_20904"

files = sorted(glob(datasets[dataset]))

dargs = []
file_chunks = zip_longest(*[iter(files)] * args.chunk_len)

for i, fc in enumerate(file_chunks):
    this_in = " ".join([f for f in fc if f is not None])
    if not this_in:
        continue
    dargs.append(
        {
            "infiles": this_in,
            "outdir": args.outdir,
        })
print(f"{len(dargs)} jobs")
proc_files = dag.layer(
    name="proc_and_merge_minbias",
    submit_description=submit_description,
    vars=dargs)

htcondor.dags.write_dag(dag, ".", args.dagname)
