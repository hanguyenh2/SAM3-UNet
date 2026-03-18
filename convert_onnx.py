import argparse
import os

import torch
import torch.onnx

from SAM3UNet import SAM3UNet

# For verification
try:
    import numpy as np
    import onnxruntime as ort

    _has_onnxruntime = True
except ImportError:
    _has_onnxruntime = False
    print("Warning: onnxruntime and numpy not found. Skipping ONNX model verification.")


def convert_pth_to_onnx(model, dummy_input, onnx_path, verbose=False):
    model.eval()
    print("Model set to evaluation mode for ONNX export.")

    try:
        # We use opset 17 as requested; it works now that complex types are removed
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["images"],
            output_names=["output"],
        )
        print(f"Model successfully exported to ONNX: {onnx_path}")
    except Exception as e:
        print(f"Error exporting model to ONNX: {e}")
        return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="path to the checkpoint of sam3-unet"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        default=True,
        help="Verify the ONNX model output against PyTorch output.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Initialize model
    model = SAM3UNet(img_size=1008).to(device)

    # 2. Load and Transform State Dict
    print(f"Loading checkpoint: {args.checkpoint}")
    state_dict = torch.load(args.checkpoint, map_location=device)

    # Logic to fix the Complex -> Real mismatch for freqs_cis
    new_state_dict = {}
    for k, v in state_dict.items():
        if "freqs_cis" in k:
            # If the checkpoint has complex tensors, convert them to real view [..., 2]
            if torch.is_complex(v):
                new_state_dict[k] = torch.view_as_real(v)
            else:
                new_state_dict[k] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=True)
    print("State dict loaded successfully with RoPE transformation.")

    # 3. Create dummy input
    dummy_input = torch.randn(1, 3, 1008, 1008).to(device)

    # 4. Define output path
    checkpoint_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    onnx_path = f"{checkpoint_name}.onnx"

    # 5. Convert
    success = convert_pth_to_onnx(model, dummy_input, onnx_path)

    # 6. Verification
    if success and args.verify and _has_onnxruntime:
        print("\n--- Verifying ONNX model output against PyTorch output ---")
        try:
            # Move dummy_input to CPU and convert to NumPy for ONNX Runtime
            dummy_input_np = dummy_input.cpu().numpy()

            # Initialize ONNX Runtime session
            # Prefer CUDA if available, otherwise fallback to CPU
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if torch.cuda.is_available()
                else ["CPUExecutionProvider"]
            )
            ort_session = ort.InferenceSession(onnx_path, providers=providers)

            # Get PyTorch model output for comparison
            with torch.no_grad():
                torch_outputs = model(dummy_input)
                # Ensure torch_outputs is a list/tuple for consistent iteration
                if not isinstance(torch_outputs, (list, tuple)):
                    torch_outputs = [torch_outputs]
                torch_outputs_np = [o.cpu().numpy() for o in torch_outputs]

            # Get ONNX model output
            ort_inputs = {ort_session.get_inputs()[0].name: dummy_input_np}
            ort_outputs = ort_session.run(None, ort_inputs)

            # Compare outputs numerically
            for i, (torch_out_np, ort_out_np) in enumerate(zip(torch_outputs_np, ort_outputs)):
                # Note: We use a slightly higher tolerance for Transformer exports
                np.testing.assert_allclose(torch_out_np, ort_out_np, rtol=1e-02, atol=1e-02)
                print(
                    f"Output {i} matched (max diff: {np.max(np.abs(torch_out_np - ort_out_np)):.2e})"
                )
            print("ONNX model verified successfully!")

        except Exception as e:
            print(f"Error during ONNX model verification: {e}")
    elif args.verify and not _has_onnxruntime:
        print("Skipping verification: onnxruntime and numpy are required but not installed.")
