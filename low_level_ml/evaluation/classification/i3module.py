import argparse
import importlib
import os
import sys

import numpy as np
import tensorflow as tf

from I3Tray import I3Tray
from icecube import dataclasses, icetray, dataio
from icecube import ml_suite

from typing import Optional, Callable, List


class I3DeepiceHexDataTransformer(ml_suite.hex.HexDataTransformer):
    """Creates a data transformer for converting an input tensor's string axis
    to a 2 axes hexagonal grid and applies normalization.

    Parameters
    ----------
    config : str or dict
        Path to yaml file or a dictionary containing `string_hex_dict`
        transformation definition.
    name : str
        Transformation name key in the config dictionary.
    string_axis : int, optional
        The axis which defines the DOM string number in the input tensor
        `x_input`.
    norm_transformations: list of callables, optional
        The features normalization list containing callables for each feature.

    Returns
    -------
    array_like
        The output tensor which is now transformed to an hexagonal grid.
        Shape: [..., 10, 10, ...] with 10, 10 inserted at the specified
        `string_axis`.
    """

    def __init__(
        self,
        config: (str, dict),
        name: str,
        string_axis: int = 1,
        norm_transformations: Optional[List[Callable]] = None,
    ):
        super(I3DeepiceHexDataTransformer, self).__init__(
            config, name, string_axis=string_axis
        )

        self._norm_transformations = norm_transformations

    def __call__(self, x_input):
        x_output = super(I3DeepiceHexDataTransformer, self).__call__(x_input)

        # Normalize features.
        if self._norm_transformations is not None:
            for i, norm_trafo in enumerate(self._norm_transformations):
                x_output[..., i] = norm_trafo(x_output[..., i])

        return x_output


def get_model(model, i3deepice_dir):
    sys.path.append(i3deepice_dir)

    model_def = importlib.import_module(f"i3deepice.models.{model}.model")

    runinfo = np.load(
        os.path.join(i3deepice_dir, f"i3deepice/models/{model}/run_info.npy"),
        allow_pickle=True,
    )[()]
    inp_shapes = runinfo["inp_shapes"]
    out_shapes = runinfo["out_shapes"]

    nn_model = model_def.model(inp_shapes, out_shapes)
    nn_model.load_weights(
        os.path.join(i3deepice_dir, f"i3deepice/models/{model}/weights.npy")
    )

    return nn_model


def get_i3deepice_trafo(config):
    def divide_100(x_input):
        return x_input / 100.0

    def divide_10000(x_input):
        return x_input / 10000.0

    # Feature normalization.
    norm_transformations = [
        divide_100,
        divide_10000,
        divide_100,
        divide_100,
        divide_100,
        divide_10000,
        divide_10000,
        divide_10000,
        divide_10000,
        divide_10000,
        divide_10000,
        divide_10000,
        divide_10000,
        divide_10000,
        divide_10000,
        divide_10000,
    ]

    i3deepice_trafo = I3DeepiceHexDataTransformer(
        config, "i3deepice_grid", norm_transformations=norm_transformations
    )

    return i3deepice_trafo


def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--files", help="files to be processed", type=str, nargs="+", required=True
    )
    parser.add_argument("--i3deepice_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--model", type=str, default="classification")
    parser.add_argument("--outfile", type=str, default=None)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parseArguments()

    nn_model = get_model(args.model, args.i3deepice_dir)

    config = os.path.join(
        os.getenv("I3_BUILD"), "ml_suite/resources/grid_transformations.yaml"
    )
    i3deepice_trafo = get_i3deepice_trafo(config)

    # Create feature extractor.
    classification_features_model = os.path.join(
        os.getenv("I3_BUILD"), f"ml_suite/resources/{args.model}_model.yaml"
    )
    eff = ml_suite.EventFeatureFactory(classification_features_model)
    event_feat_ext = eff.make_feature_extractor()

    # Create and execute I3Tray.
    tray = I3Tray()
    tray.Add("I3Reader", "reader", Filenamelist=args.files)

    # make sure that our pulses are not empty

    tray.Add(
        ml_suite.TFModelWrapper,
        "TFModelWrapper",
        nn_model=nn_model,
        event_feature_extractor=event_feat_ext,
        data_transformer=i3deepice_trafo,
        batch_size=args.batch_size,
        output_key=f"ml_suite_{args.model}",
        sub_event_stream="InIceSplit",
    )

    if args.outfile is not None:
        tray.Add("I3Writer", "EventWriter", filename=args.outfile)

    tray.Execute()
    tray.Finish()
