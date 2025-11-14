import argparse
import pathlib

import joblib
import torch
import numpy as np


def print_dict_tree(data, name="root", indent=0):
    """Recursively print a tree view of nested dictionaries."""
    pad = "  " * indent
    if isinstance(data, np.ndarray):
        print(f"{pad}{name}: ndarray shape={data.shape} dtype={data.dtype}")
    elif isinstance(data, torch.Tensor):
        print(f"{pad}{name}: Tensor shape={tuple(data.shape)} dtype={data.dtype}")
    elif isinstance(data, dict):
        print(f"{pad}{name}: dict ({len(data)})")
        for key, value in data.items():
            print_dict_tree(value, name=str(key), indent=indent + 1)
    elif isinstance(data, (list, tuple)):
        print(f"{pad}{name}: {type(data).__name__} ({len(data)})")
        for idx, value in enumerate(data):
            print_dict_tree(value, name=f"[{idx}]", indent=indent + 1)
    else:
        print(f"{pad}{name}: {type(data).__name__}")
        
def _split_pose(pose_array, body_dims=63):
    pose = torch.as_tensor(pose_array, dtype=torch.float32)
    if pose.shape[-1] < 3 + body_dims:
        raise ValueError(f"Pose dimension {pose.shape[-1]} too small for body dims {body_dims}.")
    global_orient = pose[:, :3]
    body_pose = pose[:, 3:3 + body_dims]
    return global_orient, body_pose


def _to_torch(array_like, dtype=torch.float32):
    return torch.as_tensor(array_like, dtype=dtype).clone()


def prompthmr_to_gvhmr_structure(prompthmr_results):
    """Reformat PromptHMR-style results to the GVHMR structure."""
    camera = prompthmr_results["camera"]
    people = prompthmr_results["people"]
    if not people:
        raise ValueError("No people entries found in PromptHMR results.")
    person = next(iter(people.values()))

    world = person["smplx_world"]
    cam = person["smplx_cam"]

    num_frames = world["pose"].shape[0]

    # Global (world) parameters
    global_orient_w, body_pose_w = _split_pose(world["pose"], body_dims=63)
    betas_w = _to_torch(world["shape"])
    transl_w = _to_torch(world["trans"])

    # Camera (incam) parameters
    global_orient_c, body_pose_c = _split_pose(cam["pose"], body_dims=63)
    betas_c = _to_torch(cam["shape"])
    transl_c = _to_torch(cam["trans"])

    # Intrinsics -> assume constant across frames
    focal = float(np.array(camera["img_focal"], dtype=np.float32).reshape(-1)[0])
    center = np.array(camera["img_center"], dtype=np.float32)
    if center.ndim == 0:
        cx = cy = float(center)
    else:
        cx = float(center[0])
        cy = float(center[1]) if center.size > 1 else float(center[0])
    K = torch.tensor([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]], dtype=torch.float32)
    K_fullimg = K.unsqueeze(0).repeat(num_frames, 1, 1)

    # Static confidence logits if present
    static_conf = cam.get("static_conf_logits", None)
    if static_conf is not None:
        static_conf = _to_torch(static_conf).unsqueeze(0).float()

    smpl_params_global = {
        "body_pose": body_pose_w,
        "betas": betas_w,
        "global_orient": global_orient_w,
        "transl": transl_w,
    }

    smpl_params_incam = {
        "body_pose": body_pose_c,
        "betas": betas_c,
        "global_orient": global_orient_c,
        "transl": transl_c,
    }

    net_outputs = {}
    if static_conf is not None:
        net_outputs["static_conf_logits"] = static_conf
    # Mirror essential params for compatibility (batch dimension of 1)
    net_outputs["pred_smpl_params_global"] = {
        key: value.unsqueeze(0) for key, value in smpl_params_global.items()
    }
    net_outputs["pred_smpl_params_incam"] = {
        key: value.unsqueeze(0) for key, value in smpl_params_incam.items()
    }

    return {
        "smpl_params_global": smpl_params_global,
        "smpl_params_incam": smpl_params_incam,
        "K_fullimg": K_fullimg,
        "net_outputs": net_outputs,
    }


def _load_prompthmr_results(path):
    suffix = pathlib.Path(path).suffix.lower()
    if suffix in {".pt", ".pth", ".bin"}:
        return torch.load(path)
    return joblib.load(path)


def main():
    parser = argparse.ArgumentParser(description="Convert PromptHMR results into GVHMR format.")
    parser.add_argument("input", help="Path to PromptHMR results file (.pkl or .pt).")
    parser.add_argument("output", help="Destination path for the converted GVHMR .pt file.")
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Print a tree summary of the input, optional reference, and converted outputs.",
    )
    parser.add_argument(
        "--reference",
        help="Optional GVHMR results file to inspect for comparison.",
    )
    args = parser.parse_args()

    prompthmr_results = _load_prompthmr_results(args.input)
    converted_results = prompthmr_to_gvhmr_structure(prompthmr_results)

    if args.inspect:
        print_dict_tree(prompthmr_results, name="prompthmr_results")
        if args.reference:
            reference = torch.load(args.reference)
            print_dict_tree(reference, name="reference_gvhmr")
        print_dict_tree(converted_results, name="converted_prompthmr")

    torch.save(converted_results, args.output)
    print(f"Saved GVHMR-formatted results to {args.output}")


if __name__ == "__main__":
    main()