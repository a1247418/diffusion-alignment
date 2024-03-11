#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Adapted from https://github.com/LukasMut/gLocal/blob/main/utils/evaluation/helpers.py

import copy
import itertools
import os
import pickle
import time
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from functorch import vmap

Array = np.ndarray
Tensor = torch.Tensor


def get_things_objects(data_root: str) -> Array:
    """Load name of THINGS object concepts to sort embeddings."""
    fname = "things_concepts.tsv"
    things_objects = pd.read_csv(
        os.path.join(data_root, "concepts", fname), sep="\t", encoding="utf-8"
    )
    object_names = things_objects["uniqueID"].values
    return object_names


def convert_filenames(filenames: Array) -> Array:
    """Convert binary encoded file names into strings."""
    return np.array(
        list(map(lambda f: f.decode("utf-8").split("/")[-1].split(".")[0], filenames))
    )


def load_embeddings(
    embeddings_root: str,
    module: str = "embeddings",
    sort: str = None,
    stimulus_set: str = None,
    object_names: List[str] = None,
) -> Dict[str, Array]:
    """Load Google internal embeddings and sort them according to THINGS object sorting."""

    def get_order(filenames: List[str], sorted_names: List[str]) -> Array:
        """Get correct order of file names."""
        order = np.array([np.where(filenames == n)[0][0] for n in sorted_names])
        return order

    embeddings = {}
    for f in os.scandir(embeddings_root):
        fname = f.name
        model = fname.split(".")[0]
        with open(os.path.join(embeddings_root, fname), "rb") as f:
            embedding_file = pickle.load(f)
            embedding = embedding_file[module]
            if sort:
                filenames = embedding_file["filenames"]
                filenames = convert_filenames(filenames)
                if (sort == "things" or sort == "peterson"):
                    assert object_names, "\nTo sort features according to the THINGS object names, a list (or an array) of object names is required.\n"
                    order = get_order(filenames, object_names)
                else:  # alphanumeric sorting for multi-arrangement data
                    if stimulus_set:
                        sorted_names = sorted(
                            list(
                                filter(lambda x: x.startswith(stimulus_set), filenames)
                            )
                        )
                    else:
                        sorted_names = sorted(copy.deepcopy(filenames))
                    order = get_order(filenames, sorted_names)
                embedding_sorted = embedding[order]
                embeddings[model] = embedding_sorted
            else:
                embeddings[model] = embedding
    return embeddings


def compute_dots(triplet: Tensor, pairs: List[Tuple[int]]) -> Tensor:
    return torch.tensor([triplet[i] @ triplet[j] for i, j in pairs])


def compute_distances(triplet: Tensor, pairs: List[Tuple[int]], dist: str) -> Tensor:
    if dist == "cosine":
        dist_fun = lambda u, v: 1 - F.cosine_similarity(u, v, dim=0)
    elif dist == "euclidean":
        dist_fun = lambda u, v: torch.linalg.norm(u - v, ord=2)
    elif dist == "dot":
        dist_fun = lambda u, v: -torch.dot(u, v)
    else:
        raise Exception(
            "\nDistance function other than Cosine or Euclidean distance is not yet implemented\n"
        )
    distances = torch.tensor([dist_fun(triplet[i], triplet[j]) for i, j in pairs])
    return distances


def get_predictions(
    features: Array, triplets: Array, temperature: float = 1.0, dist: str = "cosine"
) -> Tuple[Tensor, Tensor]:
    """Get the odd-one-out choices for a given model."""
    features = torch.from_numpy(features)
    indices = {0, 1, 2}
    pairs = list(itertools.combinations(indices, r=2))
    choices = torch.zeros(triplets.shape[0])
    probas = torch.zeros(triplets.shape[0], len(indices))
    print(f"\nShape of embeddings {features.shape}\n")
    for s, (i, j, k) in enumerate(triplets):
        triplet = torch.stack([features[i], features[j], features[k]])
        distances = compute_distances(triplet, pairs, dist)
        dots = compute_dots(triplet, pairs)
        if torch.unique(distances).shape[0] == 1:
            # If all distances are the same, we set the index to -1 (i.e., signifies an incorrect choice)
            choices[s] += -1
        else:
            most_sim_pair = pairs[torch.argmin(distances).item()]
            ooo_idx = indices.difference(most_sim_pair).pop()
            choices[s] += ooo_idx
        probas[s] += F.softmax(dots * temperature, dim=0)
    return choices, probas


def accuracy(choices: List[bool], target: int = 2) -> float:
    """Computes the odd-one-out triplet accuracy."""
    return round(torch.where(choices == target)[0].shape[0] / choices.shape[0], 4)


def ventropy(probabilities: Tensor) -> Tensor:
    """Computes the entropy for a batch of (discrete) probability distributions."""

    def entropy(p: Tensor) -> Tensor:
        return -(
            torch.where(p > torch.tensor(0.0), p * torch.log(p), torch.tensor(0.0))
        ).sum()

    return vmap(entropy)(probabilities)


def get_model_choices(results: pd.DataFrame) -> Array:
    models = results.model.unique()
    model_choices = np.stack(
        [results[results.model == model].choices.values[0] for model in models],
        axis=1,
    )
    return model_choices


def filter_failures(model_choices: Array, target: int = 2):
    """Filter for triplets where every model predicted differently than humans."""
    failures, choices = zip(
        *list(filter(lambda kv: target not in kv[1], enumerate(model_choices)))
    )
    return failures, np.asarray(choices)


def get_failures(results: pd.DataFrame) -> pd.DataFrame:
    model_choices = get_model_choices(results)
    failures, choices = filter_failures(model_choices)
    model_failures = pd.DataFrame(
        data=choices, index=failures, columns=results.model.unique()
    )
    return model_failures

"""
def save_pickle(data, file_path: str, model_name:str, source: str, module_type:str):
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

    try:
        for attempts in range(5):
            try:
                existing_data = np.load(file_path, allow_pickle=True)
                break
            except pickle.UnpicklingError:
                print(f"Failed to load data, trying again... ({attempts+1}/5)")
                time.sleep(np.random.randint(5, 60))
    except FileNotFoundError:
        existing_data = {}

    if source not in existing_data.keys():
        existing_data[source] = {}
    if model_name not in existing_data[source]:
        existing_data[source][model_name] = {}

    existing_data[source][model_name][module_type] = data

    with open(file_path, "wb") as f:
        pickle.dump(existing_data, f)
"""


def convert_to_filename(model_name: str, layer_name: str, source: str):
    name = model_name
    name = name.replace("diffusion_stabilityai/stable-diffusion-2-1", "sd2")
    name = name.replace("diffusion_runwayml/stable-diffusion-v1-5", "sd1")
    name = name.replace("diffusion_stabilityai/sd-turbo", "sd2t")
    name += "-" + layer_name.replace(".", "_")
    name += ".npy"
    name = os.path.join("features", source + "-" + name)
    return name


def load_features(path: str, model_name: str, source: str, module_type: str, subfolder=None,
                  per_model_path: bool = True, cc0: bool = False):
    if subfolder is not None:
        path = os.path.join(path, subfolder)
    if per_model_path:
        path = os.path.join(path, convert_to_filename(model_name, module_type, source))
    else:
        path = os.path.join(path, "model_features_per_source_cleaned.pkl")

    if cc0:
        path = path.replace(
                ".pkl", "_cc0.pkl"
        )

    with open(path, "rb") as f:
        features = pickle.load(f)

    return features


def save_pickle(data, file_path: str, model_name: str, source: str, module_type: str, per_model_path: bool = True):
    per_model_path = per_model_path and not (file_path.endswith(".npy") or file_path.endswith(".pkl"))

    if per_model_path:
        file_path = os.path.join(file_path, convert_to_filename(model_name, module_type, source))

    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

    if not per_model_path:
        try:
            for attempts in range(5):
                try:
                    existing_data = np.load(file_path, allow_pickle=True)
                    break
                except pickle.UnpicklingError:
                    print(f"Failed to load data, trying again... ({attempts+1}/5)")
                    time.sleep(np.random.randint(5, 60))
        except FileNotFoundError:
            existing_data = {}

        if source not in existing_data.keys():
            existing_data[source] = {}
        if model_name not in existing_data[source]:
            existing_data[source][model_name] = {}

        existing_data[source][model_name][module_type] = data
    else:
        existing_data = data

    # print(file_path)
    with open(file_path, "wb") as f:
        pickle.dump(existing_data, f)

