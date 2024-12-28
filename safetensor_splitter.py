import os
import json
from glob import glob
from safetensors.torch import load_file, save_file
from safetensors import safe_open
from safetensors.torch import save_file
from shutil import copy2, copytree
from collections import defaultdict
import torch
from tqdm import tqdm
import argparse

def split_dict_by_tensor_size(tensors, n):
    """Split dictionary keys into N balanced portions based on tensor sizes."""
    keys_with_sizes = [(key, tensors[key].numel()) for key in tensors]
    keys_with_sizes.sort(key=lambda x: x[1])
    total_elements = sum(size for _, size in keys_with_sizes)
    avg_elements_per_portion = total_elements / n
    result = []
    current_portion = []
    current_size = 0
    
    for key, size in keys_with_sizes:
        if current_size + size > avg_elements_per_portion and current_portion:
            result.append(current_portion)
            current_portion = [key]
            current_size = size
        else:
            current_portion.append(key)
            current_size += size
    
    if current_portion:
        result.append(current_portion)
    
    return result

def split_model(orig_model_path, output_path, split_to_n):
    os.makedirs(output_path, exist_ok=True)

    index_file = os.path.join(orig_model_path, "model.safetensors.index.json")
    with open(index_file, "r") as f:
        index_data = json.load(f)

    if "weight_map" not in index_data:
        raise ValueError("The index file does not contain a 'weight_map' entry. Ensure the file is valid.")

    weight_map = index_data["weight_map"]
    total_size = index_data.get("metadata", {}).get("total_size", 0)

    layer_groups = defaultdict(list)
    for layer, safetensor_file in weight_map.items():
        layer_groups[safetensor_file].append(layer)

    all_layers = [(layer, file) for file, layers in layer_groups.items() for layer in layers]

    new_weight_map = {}
    for file in tqdm(layer_groups.keys(), desc="Processing layers", unit="file"):
        if str(file).find("safetensors") != -1:
            tensors = {}
            with safe_open(f"{orig_model_path}/{str(file)}", framework="pt", device="cpu") as f:
                meta = f.metadata()
                for key in f.keys():
                    tensors[key] = f.get_tensor(key)
                split_keys = split_dict_by_tensor_size(tensors, split_to_n)
                for i, portion in enumerate(split_keys):
                    file_name = f"{output_path}/{file.split('.')[0]}-{i}.safetensors"
                    save_file({key: tensors[key] for key in split_keys[i]}, file_name, metadata={'format': 'pt'})
                    for key in split_keys[i]:
                        new_weight_map[key] = file_name.split('/')[-1]
    index_data['weight_map'] = new_weight_map
    with open(f"{output_path}/model.safetensors.index.json", "w") as f:
        json.dump(index_data, f)

    for item in os.listdir(orig_model_path):
        item_path = os.path.join(orig_model_path, item)
        if os.path.isfile(item_path) and not item.endswith(".safetensors") and not item == "model.safetensors.index.json":
            copy2(item_path, output_path)
        elif os.path.isdir(item_path):
            copytree(item_path, os.path.join(output_path, os.path.basename(item_path)), dirs_exist_ok=True)

def main():
    parser = argparse.ArgumentParser(description="Split a safetensors model into smaller portions based on tensor size.")
    parser.add_argument("orig_model_path", type=str, help="Path to the original model directory.")
    parser.add_argument("output_path", type=str, help="Path to save the split model.")
    parser.add_argument("split_to_n", type=int, help="Number of portions to split the model into.")
    
    args = parser.parse_args()
    
    split_model(args.orig_model_path, args.output_path, args.split_to_n)

if __name__ == "__main__":
    main()

