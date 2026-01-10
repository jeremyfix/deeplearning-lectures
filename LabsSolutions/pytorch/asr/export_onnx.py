#!/usr/bin/env python
# coding: utf-8
"""
Script to export a trained CTCModel to ONNX format.

This script loads a trained model from checkpoint and exports it to ONNX format,
which makes it easy to deploy the model across different platforms without
PyTorch dependencies.

Usage:
    python export_onnx.py <path_to_checkpoint> <output_onnx_path>

Example:
    python export_onnx.py CTCModel_9/best_model.pt CTCModel_9/best_model.onnx
"""

import sys
import pathlib
import yaml
import torch

# Local imports
from asrlab import data
from asrlab.models import deepspeech


def load_model_from_checkpoint(checkpoint_path, config_path, device="cpu"):
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to the best_model.pt file
        config_path: Path to the config.yaml file
        device: Device to load the model on (cpu or cuda)
    
    Returns:
        model: The loaded CTCModel
    """
    # Load the configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create the character map
    charmap = data.CharMap()
    
    # Create the model
    model_class = getattr(deepspeech, config['model']['class'])
    model = model_class(
        charmap=charmap,
        **config['model']['params']
    )
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    
    return model


def export_to_onnx(model, output_path, device="cpu"):
    """
    Export a CTCModel to ONNX format - the true portable format.
    
    ONNX is the universal standard that works with:
    - ONNX Runtime (lightweight, no PyTorch needed)
    - JavaScript/Node.js via ONNX.js
    - C++, C#, Java, Go, and many other languages
    - Any framework that supports ONNX (TensorFlow, etc.)
    
    Args:
        model: The CTCModel to export
        output_path: Path where to save the ONNX model
        device: Device the model is on
    """
    # Import the wrapper class
    from asrlab.models.deepspeech import CTCModelExportable
    
    # Create the exportable wrapper
    exportable_model = CTCModelExportable(model)
    exportable_model.eval()
    
    # Create dummy input with a reasonable size
    # Users can vary this at runtime with ONNX Runtime
    T, B = 100, 1
    dummy_input = torch.zeros((T, B, model.n_mels), device=device)

    try:
        print("  Exporting to ONNX (this may take a moment)...")
        
        # Export to ONNX without dynamic_axes (they cause issues with PyTorch 2.9)
        # Let PyTorch choose the appropriate opset version for best compatibility
        torch.onnx.export(
            exportable_model,
            dummy_input,
            str(output_path),
            input_names=["input"],
            output_names=["output"],
            dynamo=False, # dynamo = True fails with pytorch 2.9
            #dynamic_shapes=({0: "T", 1: "B"},),
            dynamic_axes={"input"       : {0: "T", 1: "B"},
                          "output"      : {0: "T_out", 1: "B"}},
            verbose=False,
        )
        
        print(f"✓ Model successfully exported to ONNX: {output_path}")
        print(f"\n  ✨ ONNX is the most portable format. You can now use it with:")
        print(f"    • ONNX Runtime (Python, C++, C#, Java) - LIGHTWEIGHT, no PyTorch!")
        print(f"    • ONNX.js (JavaScript/Node.js in browsers and servers)")
        print(f"    • TensorFlow (with conversion)")
        print(f"    • TensorRT (NVIDIA optimization)")
        print(f"    • CoreML (Apple devices)")
        print(f"    • Windows ML, Android ML, and many others")
        print(f"\n  ⚠️  CURRENT LIMITATION: Model fixed to shape (100, 1, 80)")
        print(f"     This was due to PyTorch 2.9 incompatibility with dynamic_axes.")
        print(f"\n  💡 WORKAROUND: To use different input sizes, you have two options:")
        print(f"\n     Option 1: Pad/reshape inputs to (100, batch_size, 80)")
        print(f"              (most practical for inference)")
        print(f"\n     Option 2: Export with a different dummy shape:")
        print(f"              Change T,B = 100,1 to your desired shape in export_onnx.py")
        print(f"              Then re-export the model")
        print(f"\n  Example with ONNX Runtime (shape 100, 1, 80):")
        print(f"    import onnxruntime as ort")
        print(f"    import numpy as np")
        print(f"    session = ort.InferenceSession('{output_path}')")
        print(f"    # Must match exported shape exactly")
        print(f"    input_array = np.random.randn(100, 1, 80).astype(np.float32)")
        print(f"    output = session.run(None, {{'input': input_array}})")
        
    except Exception as e:
        print(f"✗ ONNX export failed: {type(e).__name__}")
        print(f"  Error: {str(e)[:200]}...")
        print(f"\n  Trying fallback: JIT (TorchScript) export...")
        raise
        try:
            traced_model = torch.jit.trace(exportable_model, dummy_input)
            jit_path = str(output_path).replace(".onnx", ".jit")
            torch.jit.save(traced_model, jit_path)
            print(f"✓ Model exported to TorchScript: {jit_path}")
            print(f"\n  Note: TorchScript requires PyTorch.")
            print(f"  ONNX + ONNX Runtime is the preferred option for portability.")
        except Exception as e2:
            print(f"✗ Fallback also failed: {e2}")
            raise e



def main():
    if len(sys.argv) < 2:
        print("Usage: python export_onnx.py <path_to_checkpoint_dir> [output_onnx_path]")
        print("\nExample:")
        print("  python export_onnx.py CTCModel_9")
        print("  python export_onnx.py CTCModel_9 my_model.onnx")
        sys.exit(1)
    
    checkpoint_dir = pathlib.Path(sys.argv[1])
    
    if not checkpoint_dir.exists():
        print(f"Error: Checkpoint directory not found: {checkpoint_dir}")
        sys.exit(1)
    
    # Check for required files
    checkpoint_path = checkpoint_dir / "best_model.pt"
    config_path = checkpoint_dir / "config.yaml"
    
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    # Determine output path
    if len(sys.argv) > 2:
        output_path = pathlib.Path(sys.argv[2])
    else:
        output_path = checkpoint_dir / "best_model.onnx"
    
    # Create parent directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading model from: {checkpoint_path}")
    model = load_model_from_checkpoint(checkpoint_path, config_path)
    
    print(f"Exporting to ONNX: {output_path}")
    export_to_onnx(model, output_path)


if __name__ == "__main__":
    main()
