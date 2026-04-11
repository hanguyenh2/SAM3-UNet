import argparse
import os
import traceback

import torch
import torch.onnx

from SAM3UNet import SAM3UNet

# For verification (install with pip install onnxruntime numpy)
try:
    import numpy as np
    import onnxruntime as ort

    _has_onnxruntime = True
except ImportError:
    _has_onnxruntime = False
    print("Warning: onnxruntime and numpy not found. Skipping ONNX model verification.")


def get_replace_freqs_cis(module):
    if hasattr(module, "freqs_cis"):
        freqs_cos = module.freqs_cis.real.float()
        freqs_sin = module.freqs_cis.imag.float()
        # Replace the buffer
        module.register_buffer("freqs_cos", freqs_cos)
        module.register_buffer("freqs_sin", freqs_sin)
        del module.freqs_cis  # Remove complex version
    for child in module.children():
        get_replace_freqs_cis(child)


def convert_pth_to_onnx(model, dummy_input, onnx_path, verbose=False, dynamic_batch_size=False):
    """
    Converts a PyTorch .pth model to the ONNX format.

    Args:
        model (torch.nn.Module): The PyTorch model (loaded from .pth).
        dummy_input (torch.Tensor): A dummy input tensor with the correct shape
            for the model. This is used to trace the model's execution.
        onnx_path (str): The path where the ONNX file should be saved (e.g., "model.onnx").
        verbose (bool, optional): If True, prints detailed information during the
            conversion process. Defaults to False.
        dynamic_batch_size (bool, optional): If True, the exported ONNX model will
            support variable batch sizes. Defaults to False.
    """
    get_replace_freqs_cis(model)
    model.eval()  # Set to evaluation mode for export (good practice)
    print("Model set to evaluation mode for ONNX export.")

    input_names = ["images"]
    output_names = ["output"]  # Consider more descriptive names

    # Not recommended to use dynamic axes
    # dynamic_axes = {}
    # if dynamic_batch_size:
    #     print("Exporting ONNX model with dynamic batch size.")
    #     dynamic_axes = {
    #         "images": {0: "batch_size"},
    #         "output": {0: "batch_size"},
    #         "output_1": {0: "batch_size"},
    #         "output_2": {0: "batch_size"},
    #     }
    # If your model has other variable dimensions (e.g., variable image size),
    # you would add them here too, e.g., {'images': {0: 'batch_size', 2: 'height', 3: 'width'}}

    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            verbose=verbose,
        )
        print(f"Model successfully exported to ONNX: {onnx_path}")
    except Exception as e:
        traceback.print_exc()
        print(f"Error exporting model to ONNX: {e}")
        return False

    return True


parser = argparse.ArgumentParser()
parser.add_argument(
    "--checkpoint", type=str, required=True, help="path to the checkpoint of sam2-unet"
)
parser.add_argument(
    "--size", type=int, default=336, help="Input image size (height and width) for the model."
)
# parser.add_argument(
#     "--dynamic_batch_size", action="store_true", help="Export ONNX model with dynamic batch size."
# )
parser.add_argument(
    "--verify",
    action="store_true",
    default=True,
    help="Verify the ONNX model output against PyTorch output.",
)
args = parser.parse_args()

# 1. Load your PyTorch model from .pth
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = SAM3UNet(img_size=args.size).to(device)
model.load_state_dict(torch.load(args.checkpoint, map_location=device), strict=True)  # nosec B614

# 2. Create a dummy input tensor with the correct shape
dummy_input = torch.randn(1, 3, args.size, args.size).to(device)  # Moved to device

# 3. Define the output path for the ONNX file
checkpoint_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
onnx_path = f"{checkpoint_name}.onnx"
print(f"Attempting to convert to ONNX: {onnx_path}")

# 4. Convert the model
success = convert_pth_to_onnx(
    model,
    dummy_input,
    onnx_path,
    verbose=False,  # dynamic_batch_size=args.dynamic_batch_size
)

# 5. Verify the ONNX model (Optional but highly recommended)
if success and args.verify and _has_onnxruntime:
    print("\n--- Verifying ONNX model output against PyTorch output ---")
    try:
        # Move dummy_input to CPU and convert to NumPy for ONNX Runtime
        dummy_input_np = dummy_input.cpu().numpy()

        # Get PyTorch model output for comparison
        with torch.no_grad():
            torch_outputs = model(dummy_input)
            # Ensure torch_outputs is a list/tuple for consistent iteration
            if not isinstance(torch_outputs, (list, tuple)):
                torch_outputs = [torch_outputs]
            torch_outputs_np = [o.cpu().numpy() for o in torch_outputs]

        for _ in range(1):
            # Initialize ONNX Runtime session
            # Prefer CUDA if available, otherwise fallback to CPU
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if torch.cuda.is_available()
                else ["CPUExecutionProvider"]
            )
            ort_session = ort.InferenceSession(onnx_path, providers=providers)

            # Get ONNX model output
            ort_inputs = {ort_session.get_inputs()[0].name: dummy_input_np}
            ort_outputs = ort_session.run(None, ort_inputs)  # None for all outputs

            # Compare outputs numerically
            for i, (torch_out_np, ort_out_np) in enumerate(zip(torch_outputs_np, ort_outputs)):
                # Use allclose for floating-point comparison
                assert not np.any(np.isnan(ort_out_np)), "Array should not contain NaNs"
                np.testing.assert_allclose(torch_out_np, ort_out_np, rtol=1, atol=1)
                # print(torch_out_np, ort_out_np)
                print(
                    f"Output {i} matched between PyTorch and ONNX (max diff: {np.max(np.abs(torch_out_np - ort_out_np)):.2e})"
                )
            print("ONNX model verified successfully!")

    except Exception as e:
        print(f"Error during ONNX model verification: {e}")
elif args.verify and not _has_onnxruntime:
    print("Skipping verification: onnxruntime and numpy are required but not installed.")
