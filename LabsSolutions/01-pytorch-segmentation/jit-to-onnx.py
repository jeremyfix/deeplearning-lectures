#!/usr/bin/env python3

"""
    This script belongs to the lab work on semantic segmenation of the
    deep learning lectures https://github.com/jeremyfix/deeplearning-lectures
    Copyright (C) 2022 Jeremy Fix

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

# This scripts allows for loading a module exported with torch.jit to
# the more generic onnx format

# Standard imports
import argparse
import pathlib

# External imports
import torch
import onnxruntime as ort
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--modelpath", type=pathlib.Path, required=True)
parser.add_argument(
    "--input_size",
    nargs="+",
    type=int,
    required=True,
    help="The input tensor size in the order channel x height x width",
)
parser.add_argument("--exportpath", type=str, default="model.onnx")
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda") if use_cuda else torch.device("cpu")
model = torch.jit.load(args.modelpath, map_location=device)
model.eval()

export_input_size = (1,) + tuple(args.input_size)

dummy_input = torch.rand(export_input_size, device=device)

# Forward propagate through the JIT traced model
print("Forward propagation through the JIT model")
with torch.no_grad():
    dummy_output = model(dummy_input)

print(f"ONNX export to {args.exportpath}")
torch.onnx.export(
    model,
    dummy_input,
    args.exportpath,
    verbose=True,
    opset_version=12,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
)  # At least 11 is required otherwise it seems nn.UpSample is not correctly handled

# Test the export

print("Testing inference with ORT")
providers = []
if use_cuda:
    providers.append("CUDAExecutionProvider")
providers.append("CPUExecutionProvider")

inference_session = ort.InferenceSession(args.exportpath, providers=providers)

model = lambda torch_X: inference_session.run(
    None, {inference_session.get_inputs()[0].name: torch_X.cpu().numpy()}
)[0]
onnx_output = model(dummy_input)

print(
    f"Are forward propagation through JIT model and ONNX export allclose ? {np.allclose(onnx_output,  dummy_output.cpu().numpy())}"
)
diff = np.abs(onnx_output - dummy_output.cpu().numpy())
print(f"Bounds for the diff : [{diff.min()}, {diff.max()}]")
