import os
import torch
import torch.nn.functional as F
import h5py
from tqdm import tqdm
from glob import glob
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
import argparse
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.CHIEF import CHIEF
from pathlib import Path
import pandas as pd
import numpy as np
import json
import yaml

# ---------------------- Model Loading ---------------------- #
def load_chief_model(device):
    """Load the CHIEF model for attention extraction."""
    model = CHIEF(size_arg="small", dropout=True, n_classes=2)
    chief_weights_path = os.path.join("model_weights", "CHIEF_pretraining.pth")
    td = torch.load(chief_weights_path, map_location=device)
    if "organ_embedding" in td:
        del td["organ_embedding"]
    model.load_state_dict(td, strict=True)
    model.eval().to(device)
    return model

def load_patch_feats(h5_path, device):
    """
    Load patch features from an HDF5 file.
    Args:
        h5_path (str): Path to the HDF5 file containing patch features.
        device (str): Device to load the features onto (e.g., "cuda" or "cpu").
    Returns:                                
        feats (torch.Tensor): Loaded patch features as a PyTorch tensor.
        coords (np.ndarray): Coordinates associated with the patch features.
    """
    if not os.path.exists(h5_path):
        tqdm.write(f"File {h5_path} does not exist, skipping")
        return None
    with h5py.File(h5_path, "r") as f:
        feats = f["feats"][:]
        feats = torch.tensor(feats).to(device)
        coords = np.array(f["coords"][:])
    return feats, coords

def match_coords(feats_w, feats_a, coords_w, coords_a):
    """
    Match and extract features whose corresponding coordinates are identical in two sets.

    It uses np.intersect1d to compute the intersection (in sorted order)
    of the coordinate arrays, and returns the features accordingly.

    Parameters:
        feats_w (np.ndarray): Feature array for weighted patches.
        feats_a (np.ndarray): Feature array for auxiliary patches.
        coords_w (np.ndarray): Coordinates for feats_w with shape (N, D).
        coords_a (np.ndarray): Coordinates for feats_a with shape (M, D).

    Returns:
        tuple: (matched_feats_w, matched_feats_a) where the i-th entry in both arrays corresponds 
               to the same common coordinate.

    Raises:
        ValueError: If no common coordinates are found.
    """
    dt = np.dtype((np.void, coords_w.dtype.itemsize * coords_w.shape[1]))
    coords_w_view = np.ascontiguousarray(coords_w).view(dt).ravel()
    coords_a_view = np.ascontiguousarray(coords_a).view(dt).ravel()

    common, idx_w, idx_a = np.intersect1d(coords_w_view, coords_a_view, return_indices=True)
    if len(common) == 0:
        raise ValueError("No matching coordinates found")
    
    return feats_w[idx_w], feats_a[idx_a]

def get_eagle_feats(model, patch_feats_w, patch_feats_a, top_k=25):
    """
    Compute EAGLE features by aggregating patch features using attention scores from the model.
    This function takes patch features and processes them with the given model to extract
    attention scores. If a top_k value is provided, it selects the top_k patches based on
    the attention scores, applies a softmax weighting over these scores, and computes a weighted
    sum of the corresponding patch features. Otherwise, it directly aggregates all patch
    features with the raw attention scores.
    Parameters:
        model (torch.nn.Module): The neural network model used to compute attention scores.
                                 It should accept patch_feats_w as input and support a "get_attention"
                                 keyword argument.
        patch_feats_w (torch.Tensor): The input patch features that are processed by the model to obtain
                                      attention scores.
        patch_feats_a (torch.Tensor): The patch features used for the final feature aggregation.
        top_k (int, optional): The number of top patches (based on attention score) to use for feature
                               aggregation.
    Returns:
        torch.Tensor: The aggregated EAGLE features as a 1D tensor.
    """
    with torch.inference_mode():
        A = model(patch_feats_w)["attention_raw"]
        # A.shape: (1,num_patches)
        if top_k:
            if A.size(-1) < top_k:
                top_k = A.size(-1)
            top_k_indices = torch.topk(A, top_k, dim=-1).indices  # (1,top_k)
            # Gather corresponding features from patch_feats_a
            top_k_x = patch_feats_a.gather(1, top_k_indices.unsqueeze(-1).expand(-1, -1, patch_feats_a.size(-1)))
            eagle_feats = top_k_x.mean(dim=1)
        else:
            eagle_feats = torch.bmm(A.unsqueeze(0), patch_feats_a).squeeze(1)
    return eagle_feats.squeeze(0)

def save_chunk(
    data_dict,
    chunk_index,
    output_dir,
    output_file,
    model_name,
    top_k,
    dtype,
    weighting_fm,
    aggregation_fm,
    microns,
):
    """Helper: save one chunk of features to <output_file>_chunk<chunk_index>.h5"""
    if chunk_index > 0:
        if chunk_index == 1:
            os.rename(
                os.path.join(output_dir, output_file),
                os.path.join(output_dir, f"{output_file.split('.h5')[0]}_0.h5"),
            )
        chunk_file = os.path.join(
            output_dir, f"{output_file.split('.h5')[0]}_{chunk_index}.h5"
        )
    else:
        chunk_file = os.path.join(output_dir, output_file)
    os.makedirs(os.path.dirname(chunk_file), exist_ok=True)
    with h5py.File(chunk_file, "w") as f:
        for key, data in data_dict.items():
            f.create_dataset(key, data=data["feats"])
        f.attrs.update({
            "extractor": model_name,
            "top_k": top_k if top_k else "None",
            "dtype": str(dtype),
            "weighting_FM": weighting_fm,
            "aggregation_FM": aggregation_fm,
            "microns": microns,
        })
    tqdm.write(f"Saved chunk {chunk_index} to {chunk_file}")

def get_pat_embs(
    model,
    output_dir,
    feat_dir_w,
    feat_dir_a=None,
    output_file="eagle_feats.h5",
    model_name="EAGLE",
    slide_table_path=None,
    device="cuda",
    dtype=torch.float32,
    top_k=25,
    weighting_fm="chief-CTP",
    aggregation_fm="Virchow2",
    microns=256,
    chunk_size=None  # Enable chunking if provided
):
    """
    Extract patient-level features from slide-level feature files and save them into HDF5 files.
    Supports saving in chunks if `chunk_size` is provided.
    Uses match_coords only when auxiliary features are provided and weighting_fm != aggregation_fm.
    """
    slide_table = pd.read_csv(slide_table_path)
    patient_groups = slide_table.groupby("PATIENT")
    pat_dict = {}

    do_match = (feat_dir_a is not None) and (weighting_fm != aggregation_fm)
    if do_match:
        print("Using match_coords for patient-level extraction (weighting_fm != aggregation_fm).")
    else:
        print("Skipping match_coords for patient-level extraction (using identical features or no auxiliary features).")

    chunk_index = 0
    chunk_patients = []

    for patient_id, group in tqdm(patient_groups, leave=False):
        all_feats_list_w = []
        all_feats_list_a = []

        for _, row in group.iterrows():
            slide_filename = row["FILENAME"]
            h5_path_w = os.path.join(feat_dir_w, slide_filename)
            feats_w, coords_w = load_patch_feats(h5_path_w, device)
            if feats_w is None:
                continue
            if feat_dir_a:
                h5_path_a = os.path.join(feat_dir_a, slide_filename)
                feats_a, coords_a = load_patch_feats(h5_path_a, device)
            else:
                feats_a, coords_a = feats_w, coords_w

            if feats_a is None:
                continue

            if do_match:
                try:
                    feats_w, feats_a = match_coords(feats_w, feats_a, coords_w, coords_a)
                except ValueError as e:
                    tqdm.write(f"Patient {patient_id}, slide {slide_filename}: {str(e)}")
                    continue

            all_feats_list_w.append(feats_w)
            all_feats_list_a.append(feats_a)

        if all_feats_list_w:
            all_feats_cat_w = torch.cat(all_feats_list_w, dim=0).unsqueeze(0)
            all_feats_cat_a = torch.cat(all_feats_list_a, dim=0).unsqueeze(0)
            assert all_feats_cat_w.ndim == 3, f"Expected 3D tensor, got {all_feats_cat_w.ndim}"
            assert all_feats_cat_a.ndim == 3, f"Expected 3D tensor, got {all_feats_cat_a.ndim}"
            assert all_feats_cat_w.shape[1] == all_feats_cat_a.shape[1], (
                f"Expected same number of tiles, got {all_feats_cat_w.shape[1]} and {all_feats_cat_a.shape[1]}"
            )
            patient_feats = get_eagle_feats(model, all_feats_cat_w.to(dtype), all_feats_cat_a.to(dtype), top_k=top_k)
            pat_dict[patient_id] = {
                "feats": patient_feats.to(torch.float32).detach().squeeze().cpu().numpy(),
            }
            chunk_patients.append(patient_id)

        if chunk_size and len(chunk_patients) >= chunk_size:
            save_chunk(
                pat_dict,
                chunk_index,
                output_dir,
                output_file,
                model_name,
                top_k,
                dtype,
                weighting_fm,
                aggregation_fm,
                microns,
            )
            pat_dict.clear()
            chunk_patients.clear()
            chunk_index += 1

    if pat_dict:  # final leftover
        save_chunk(
            pat_dict,
            chunk_index,
            output_dir,
            output_file,
            model_name,
            top_k,
            dtype,
            weighting_fm,
            aggregation_fm,
            microns,
        )

    metadata = {
        "extractor": model_name,
        "top_k": top_k if top_k else "None",
        "dtype": str(dtype),
        "weighting_FM": weighting_fm,
        "aggregation_FM": aggregation_fm,
        "microns": microns,
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as json_file:
        json.dump(metadata, json_file, indent=4)

def get_slide_embs(
    model,
    output_dir,
    feat_dir_w,
    feat_dir_a=None,
    output_file="eagle_feats.h5",
    model_name="EAGLE",
    device="cuda",
    dtype=torch.float32,
    top_k=25,
    weighting_fm="chief-CTP",
    aggregation_fm="Virchow2",
    microns=256,
    chunk_size=None  # Enable chunking if provided
):
    """
    Generates slide-level features from tile embeddings and saves them in HDF5 files (in chunks if chunk_size is provided) 
    along with a metadata JSON file.
    Uses match_coords only when auxiliary features are provided and weighting_fm != aggregation_fm.
    """
    slide_dict = {}
    chunk_index = 0
    chunk_slides = []

    tile_emb_paths_w = glob(f"{feat_dir_w}/**/*.h5", recursive=True)
    if feat_dir_a is not None:
        tile_emb_paths_a = glob(f"{feat_dir_a}/**/*.h5", recursive=True)
    else:
        tile_emb_paths_a = tile_emb_paths_w

    assert len(tile_emb_paths_w) == len(tile_emb_paths_a), (
        f"Expected same number of files, got {len(tile_emb_paths_w)} and {len(tile_emb_paths_a)}"
    )

    do_match = (feat_dir_a is not None) and (weighting_fm != aggregation_fm)
    if do_match:
        print("Using match_coords for slide-level extraction (weighting_fm != aggregation_fm).")
    else:
        print("Skipping match_coords for slide-level extraction (using identical features or no auxiliary features).")

    for tile_emb_path_w, tile_emb_path_a in zip(tqdm(tile_emb_paths_w), tile_emb_paths_a):
        slide_name = Path(tile_emb_path_w).stem
        feats_w, coords_w = load_patch_feats(tile_emb_path_w, device)
        if feats_w is None:
            continue
        if feat_dir_a:
            tile_emb_path_a = os.path.join(feat_dir_a, f"{slide_name}.h5")
            feats_a, coords_a = load_patch_feats(tile_emb_path_a, device)
        else:
            feats_a, coords_a = feats_w, coords_w
        if feats_a is None:
            continue

        if do_match:
            try:
                feats_w, feats_a = match_coords(feats_w, feats_a, coords_w, coords_a)
            except ValueError as e:
                tqdm.write(f"Slide {slide_name}: {str(e)}")
                continue

        tile_embs_w = feats_w.unsqueeze(0)
        tile_embs_a = feats_a.unsqueeze(0)
        assert tile_embs_w.ndim == 3, f"Expected 3D tensor, got {tile_embs_w.ndim}"
        assert tile_embs_a.ndim == 3, f"Expected 3D tensor, got {tile_embs_a.ndim}"
        assert tile_embs_w.shape[1] == tile_embs_a.shape[1], (
            f"Expected same number of tiles, got {tile_embs_w.shape[1]} and {tile_embs_a.shape[1]}"
        )

        slide_feats = get_eagle_feats(model, tile_embs_w.to(dtype), tile_embs_a.to(dtype), top_k=top_k)
        slide_dict[slide_name] = {
            "feats": slide_feats.to(torch.float32).detach().cpu().numpy(),
            "extractor": model_name,
        }
        chunk_slides.append(slide_name)

        if chunk_size and len(chunk_slides) >= chunk_size:
            save_chunk(
                slide_dict,
                chunk_index,
                output_dir,
                output_file,
                model_name,
                top_k,
                dtype,
                weighting_fm,
                aggregation_fm,
                microns,
            )
            slide_dict.clear()
            chunk_slides.clear()
            chunk_index += 1

    if slide_dict:
        if chunk_size:
            save_chunk(
                slide_dict,
                chunk_index,
                output_dir,
                output_file,
                model_name,
                top_k,
                dtype,
                weighting_fm,
                aggregation_fm,
                microns,
            )
            tqdm.write(f"Finished extraction, saved chunk {chunk_index} to {output_dir}")
        else:
            output_path = os.path.join(output_dir, output_file)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with h5py.File(output_path, "w") as f:
                for slide_name, data in slide_dict.items():
                    f.create_dataset(f"{slide_name}", data=data["feats"])
                f.attrs["extractor"] = model_name
                f.attrs["top_k"] = top_k if top_k else "None"
                f.attrs["dtype"] = str(dtype)
                f.attrs["weighting_FM"] = weighting_fm
                f.attrs["aggregation_FM"] = aggregation_fm
                f.attrs["microns"] = microns
            tqdm.write(f"Finished extraction, saved to {output_path}")

    metadata = {
        "extractor": model_name,
        "top_k": top_k if top_k else "None",
        "dtype": str(dtype),
        "weighting_FM": weighting_fm,
        "aggregation_FM": aggregation_fm,
        "microns": microns,
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as json_file:
        json.dump(metadata, json_file, indent=4)

def main():
    """
    Main function for extracting slide or patient embeddings using the EAGLE model.
    
    This function parses command-line arguments (and optionally a YAML configuration file)
    to set up the extraction parameters. It supports two modes:
      - Patient-level embeddings (if a slide table is provided)
      - Slide-level embeddings (if not)
    """
    parser = argparse.ArgumentParser(
        description="Extract slide/patient embeddings using the EAGLE model"
    )
    parser.add_argument("-c", "--config", type=str,
                        help="Path to a YAML configuration file", default=None)
    parser.add_argument("-o", "--output_dir", type=str, required=True,
                        help="Directory to save extracted features")
    parser.add_argument("-f", "--feat_dir", type=str, required=True,
                        help="Directory containing tile feature files")
    parser.add_argument("-g", "--feat_dir_a", type=str, required=False, default=None,
                        help="Directory containing tile feature files for aggregation")
    parser.add_argument("-k", "--top_k", type=int, required=False, default=25,
                        help="Top k tiles to use for slide/patient embedding")
    parser.add_argument("-m", "--model_name", type=str, required=False, default="EAGLE",
                        help="Model name")
    parser.add_argument("-p", "--patch_encoder", type=str, required=False, default="chief-CTP",
                        help="Patch encoder name")
    parser.add_argument("-a", "--patch_encoder_a", type=str, required=False, default="Virchow2",
                        help="Patch encoder name used for aggregation")
    parser.add_argument("-e", "--h5_name", type=str, required=False, default="eagle_feats.h5",
                        help="Output HDF5 file name")
    parser.add_argument("-r", "--microns", type=int, required=False, default=256,
                        help="Microns per patch used for extraction")
    parser.add_argument("-s", "--slide_table", type=str, required=False,
                        help="Slide table path (for patient-level extraction)")
    parser.add_argument("-z", "--chunk_size", type=int, required=False, default=None,
                        help="Maximum number of slides or patients per HDF5 file (for chunked saving)")
    args = parser.parse_args()
    
    if args.config is not None:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        config = config.get("extract_feats", {})
        args.output_dir = config.get("output_dir", args.output_dir)
        args.feat_dir = config.get("feat_dir", args.feat_dir)
        args.top_k = config.get("top_k", args.top_k)
        args.feat_dir_a = config.get("feat_dir_a", args.feat_dir_a)
        args.model_name = config.get("model_name", args.model_name)
        args.patch_encoder = config.get("patch_encoder", args.patch_encoder)
        args.patch_encoder_a = config.get("patch_encoder_a", args.patch_encoder_a)
        args.h5_name = config.get("h5_name", args.h5_name)
        args.microns = config.get("microns", args.microns)
        args.slide_table = config.get("slide_table", args.slide_table)
        args.chunk_size = config.get("chunk_size", args.chunk_size)
    
    print(f"Using configuration: {args}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_chief_model(device)
    model.eval()
    dtype = torch.float32

    if args.slide_table:
        # Patient-level embeddings
        get_pat_embs(
            model,
            args.output_dir,
            args.feat_dir,
            args.feat_dir_a,
            args.h5_name,
            args.model_name,
            args.slide_table,
            device,
            dtype=dtype,
            top_k=args.top_k,
            weighting_fm=args.patch_encoder,
            aggregation_fm=args.patch_encoder_a,
            microns=args.microns,
            chunk_size=args.chunk_size,      
        )
    else:
        # Slide-level embeddings
        get_slide_embs(
            model,
            args.output_dir,
            args.feat_dir,
            args.feat_dir_a,
            args.h5_name,
            args.model_name,
            device=device,
            dtype=dtype,
            top_k=args.top_k,
            weighting_fm=args.patch_encoder,
            aggregation_fm=args.patch_encoder_a,
            microns=args.microns,
            chunk_size=args.chunk_size,     
        )
        
if __name__ == "__main__":
    main()