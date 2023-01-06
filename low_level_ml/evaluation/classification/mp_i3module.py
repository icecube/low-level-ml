# Run the `Theo Classifier` on a single GPU with multiple CPU IceTray workers for
# much more efficient GPU resource usage.

# This only results in a speedup when run on at least as many files as there are
# CPU workers (set via the `--num-workers` argument) since each worker opens one
# file at a time.

# The optimal settings highly depend on the system used, play with the number of
# workers and the batch size until the GPU utilization (monitor with
# `nvidia-smi` command) does not increase anymore and the tqdm iterations/second
# indicator is maximized.
# 2023 - Alexander Harnisch
import glob
import multiprocessing as mp
import os
from argparse import ArgumentError
from multiprocessing import RawArray
from multiprocessing.connection import Connection, wait

import click
import numpy as np
from I3Tray import I3Tray
from icecube import dataclasses, dataio, icetray, ml_suite
from tqdm import tqdm

from i3module import get_model, get_i3deepice_trafo


@click.command()
@click.argument("input_patterns", required=True, nargs=-1)
@click.option(
    "--i3deepice-dir",
    type=click.Path(exists=True),
    help="The i3deepice directory/path.",
)
@click.option(
    "--i3deepice-model",
    default="classification",
    help="The name of the i3deepice model to use.",
)
@click.option(
    "--ml-suite-config",
    default="theo_dnn_classification",
    help="Name of the ML Suite configuration file in ml_suite/resources/.",
)
@click.option(
    "--dest",
    default="./",
    type=click.Path(exists=True),
    help="File path to output folder.",
)
@click.option("--num-workers", default=10)
@click.option("--batch-size", default=1024)
@click.option("--subeventstream", "-s", default="InIceSplit")
@click.option("--skip-existing/--overwrite-existing", default=True)
def main(
    input_patterns,
    i3deepice_dir,
    i3deepice_model,
    ml_suite_config,
    dest,
    batch_size,
    num_workers,
    subeventstream,
    skip_existing,
):
    # Add slash to dest, if not already there
    if dest[-1] != "/":
        dest += "/"

    # Load ML Suite config file path
    maybe_path = os.path.join(
        os.getenv("I3_BUILD"), f"ml_suite/resources/{ml_suite_config}_model.yaml"
    )
    if not os.path.exists(maybe_path):
        if not os.path.exists(ml_suite_config):
            raise ValueError(f"Could not load config {ml_suite_config}")
        else:
            maybe_path = ml_suite_config

    classification_features_model = maybe_path

    # Initialize multiprocessing file queue
    file_queue = mp.Queue()

    # Gather all input files and add them to the queue
    for input_pattern in input_patterns:
        pattern_files = sorted(glob.glob(input_pattern))
        if len(pattern_files) == 0:
            raise FileNotFoundError(f"Could not find any files for pattern {input_pattern}.")
        for file in pattern_files:
            file_queue.put(file)

    # Load the model (only in this parent/server process)
    print("Loading the model...")
    nn_model, output_names = get_model(i3deepice_model, i3deepice_dir)

    config = os.path.join(
        os.getenv("I3_BUILD"), "ml_suite/resources/grid_transformations.yaml"
    )

    # Gather worker arguments in dictionary
    worker_args = {
        "dest": dest,
        "batch_size": batch_size,
        "subeventstream": subeventstream,
        "skip_existing": skip_existing,
        "ml_suite_config": classification_features_model,
        "i3deepice_model": i3deepice_model,
        "output_names": output_names,
    }

    # Load the transformer
    transformer = get_i3deepice_trafo(config)

    # Infer batch shape
    # The number of features is technically dynamic, but not really since it
    # must be the same as the length of `norm_transformations` in
    # `get_i3deepice_trafo`, which is not dynamic. So it is effectively
    # hardcoded to 16.
    dummy = np.zeros((batch_size, 86, 60, 16))
    batch = transformer(dummy)
    batch_shape = batch.shape
    batch_dtype = np.ctypeslib.as_ctypes_type(batch.dtype)
    del dummy, batch
    worker_args["batch_shape"] = batch_shape

    processes = []
    connections = []
    shm_arrays = []
    print(f"Starting {num_workers} worker processes")
    for rank in range(num_workers):
        parent_conn, child_conn = mp.Pipe()
        shm_array = RawArray(batch_dtype, int(np.prod(batch_shape)))
        process = mp.Process(
            target=icetray_worker,
            args=(rank, child_conn, shm_array, file_queue, transformer, worker_args),
            daemon=True,
        )
        process.start()
        processes.append(process)
        connections.append(parent_conn)
        shm_arrays.append(np.frombuffer(shm_array, dtype=np.dtype(shm_array)).reshape(batch_shape))
        file_queue.put(False)  # One poison pill for each worker

    print("Starting prediction server")
    with tqdm() as pbar:
        while connections:
            for conn in wait(connections):
                msg = conn.recv()
                if type(msg) is not bool:
                    conn.send(nn_model(shm_arrays[msg]).numpy())
                    pbar.update(1)
                else:
                    connections.remove(conn)
    for p in processes:
        p.join()


def icetray_worker(
    rank: int,
    conn: Connection,
    shm_array: RawArray,
    file_queue: mp.Queue,
    transformer,
    worker_args: dict,
):
    # Create a numpy array *view* of the shared memory batch array
    x_np = np.frombuffer(shm_array, dtype=np.dtype(shm_array)).reshape(worker_args["batch_shape"])

    # Create feature extractor
    eff = ml_suite.EventFeatureFactory(worker_args["ml_suite_config"])
    event_feat_ext = eff.make_feature_extractor()

    def call_model(x):
        batch = transformer(x)
        np.copyto(x_np[: len(batch)], batch)
        conn.send(rank)
        return conn.recv()[: len(batch)]

    while True:
        in_file = file_queue.get()
        if not in_file:  # Poison pill, all tasks done
            break

        i3_out_file = worker_args["dest"] + os.path.basename(in_file)
        if not os.path.exists(i3_out_file) or not worker_args["skip_existing"]:
            tray = I3Tray()
            tray.AddModule("I3Reader", "reader", Filenamelist=[in_file])

            tray.AddModule(
                ml_suite.ModelWrapper,
                "ModelWrapper",
                nn_model=call_model,
                event_feature_extractor=event_feat_ext,
                batch_size=worker_args["batch_size"],
                output_key=f"ml_suite_{worker_args['i3deepice_model']}",
                output_names=worker_args["output_names"],
                write_runtime_info=False,
                sub_event_stream=worker_args["subeventstream"],
            )
            tray.AddModule(
                "I3Writer",
                "EventWriter",
                DropOrphanStreams=[icetray.I3Frame.DAQ],
                filename=i3_out_file,
            )
            tray.Execute()
            tray.Finish()
            tray.PrintUsage()
        elif os.path.exists(i3_out_file):
            print(i3_out_file, "already exists, skipping")

    conn.send(False)  # Termination signal
    conn.close()


if __name__ == "__main__":
    main()
