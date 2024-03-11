import argparse
import os
import numpy as np
from torchvision.transforms import ToTensor, Compose, Resize
from sklearn.decomposition import PCA

import helpers
from embedding import embed
from things import THINGSBehavior


def parseargs():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    aa("--data_root", type=str, help="path/to/things")
    aa("--out_path", type=str, help="path/to/things")
    aa("--model", type=str, default="diffusion_stabilityai/stable-diffusion-2-1_20")
    aa(
        "--module",
        type=str,
        default=["down_blocks.3.resnets.1", "mid_block", "up_blocks.1.resnets.1"],
        nargs="+",
    )
    args = parser.parse_args()
    return args


def load_dataset(data_dir: str, transform=None, cc0=False):
    dataset = THINGSBehavior(
        root=data_dir, aligned=False, download=False, transform=transform, cc0=cc0
    )
    return dataset


def reduce_align(
    features, things_dir: str, reduction: str, pca_dim=None, n_triplets=100000
):
    if reduction == "none":
        features_reduced = np.reshape(features, (features.shape[0], -1))
    elif reduction == "max":
        # features_reduced = torch.max(features, dim=[-2,-1]).numpy()
        features_reduced = np.max(features, axis=(-2, -1))
    elif reduction == "avg":
        # features_reduced = torch.mean(features, dim=[-2,-1]).numpy()
        features_reduced = np.mean(features, axis=(-2, -1))
    elif reduction == "pca":
        pca = PCA(n_components=pca_dim)
        features_reduced = np.reshape(features, (features.shape[0], -1))
        features_reduced = pca.fit_transform(features_reduced)

    transform = Compose([Resize(512), ToTensor()])
    dataset = load_dataset(things_dir, transform=transform, cc0=False)
    triplets = dataset.get_triplets()

    # get a reproducible random subset of 100k triplets
    triplets = np.random.RandomState(42).permutation(triplets)[:n_triplets]

    choices, probas = helpers.get_predictions(
        features_reduced, triplets, temperature=1.0, dist="cosine"
    )
    acc = helpers.accuracy(choices)
    print("Zero-shot odd-one-out accuracy:", acc)

    return acc


if __name__ == "__main__":
    # parse arguments
    args = parseargs()

    for module in args.module:
        print(f"Module: {module}")
        # 1. embed
        features = embed(
            path_to_embeddings=os.path.join(args.out_path, "unreduced_features.pkl"),
            path_to_things=args.data_root,
            path_to_model_dict=None,
            model_name=args.model,
            source="diffusion",
            module_type=module,
            cc0=False,
            overwrite=False,
            pretrained=True,
            pool=False,
            path_to_caption_dict=None,
            save=True,
        )
        print(features.shape)

        # 2. reduce + calculate alignment
        for reduction in ["avg", "max", "none", "pca"]:
            print("\n-------------------------------------------")
            print(f"Reduction: {reduction}")

            pca_dims = [None]
            if reduction == "pca":
                pca_dims = [
                    20,
                    50,
                    100,
                    150,
                    200,
                    500,
                    1000,
                    min(1854, features.shape[-1]),
                ]

            for pca_dim in pca_dims:
                print(f"PCA dim: {pca_dim}")
                out_file_path = os.path.join(args.out_path, f"reduced_{reduction}")
                if not os.path.exists(out_file_path):
                    os.makedirs(out_file_path)

                acc = reduce_align(features, args.data_root, reduction, pca_dim=pca_dim)
                # save results
                out_file_path = os.path.join(
                    out_file_path, f"results_{reduction}_{pca_dim}.npy"
                )
                helpers.save_pickle(
                    acc,
                    file_path=out_file_path,
                    model_name=args.model,
                    source="diffusion",
                    module_type=module,
                    per_model_path=False,
                )
    print("Done.")
