import json
import numpy as np
import torch
from typing import Optional
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor
from tqdm import tqdm
from thingsvision import get_extractor
from thingsvision.utils.data import DataLoader as DL

from things import THINGSBehavior
from diffusion_encoders import StableDiffusionEncoder, SD3TransformerBlockEncoder

from helpers import save_pickle


def embed(
    path_to_embeddings: str,
    path_to_things: str,
    path_to_model_dict: Optional[str],
    model_name: str,
    source: str,
    module_type: str,
    cc0: bool = False,
    overwrite: bool = False,
    pretrained: bool = True,
    pool: bool = False,
    path_to_caption_dict: Optional[dict] = None,
    save: bool = True,
):
    print("Embedding", source, model_name, module_type, "cc0 =", cc0, "pretrained = ", pretrained)
    if path_to_model_dict is None:
        print("No model dict path provided. Interpreting module type as module name.")

    # Resolve extractor params
    if source == "diffusion":
        parts = model_name.split("_")
        name = parts[0]
        ckpt = parts[1]
        noise = int(parts[2])
        model_params = dict(ckpt=ckpt, noise=noise)
    elif model_name.startswith("OpenCLIP"):
        parts = model_name.split("_")
        name = parts[0]
        variant = parts[1]
        data = "_".join(parts[2:])
        model_params = dict(variant=variant, dataset=data)
    elif model_name.startswith("clip"):
        name, variant = model_name.split("_")
        model_params = dict(variant=variant)
    else:
        name = model_name
        model_params = None

    # Load from file
    error = False
    if not overwrite:
        try:
            features = np.load(path_to_embeddings, allow_pickle=True)[source][
                (model_name + "-untrained") if not pretrained else model_name
            ][module_type]
            print("Loaded embeddings from file. Skipping.")
        except Exception as e:
            print("Could not load embeddings from file:", e)
            error = True

    # Calculate from data
    if error or overwrite:
        if path_to_caption_dict is not None and path_to_caption_dict != "":
            caption_dict = np.load(path_to_caption_dict, allow_pickle=True)
            try:
                caption_dict = caption_dict[()]
            except KeyError:
                pass
        else:
            caption_dict = None

        if source == "diffusion":
            # Diffusion models
            eval_txt_enc = model_name.startswith("text")
            eval_txt_enc_id = 1 if model_name.startswith("textfirst") else -1
            conditional = model_name.startswith("conditional") or eval_txt_enc
            captions = model_name.startswith("conditionalcapt") or model_name.startswith("textlastcapt")
            optim = model_name.startswith("optim") or model_name.startswith("conditionaloptim")
            n_to_avg = 5 if "avg5" in model_name else 1

            if captions:
                assert caption_dict is not None, "To use captions, a caption dict must be provided."

            transform = Compose([Resize(512), ToTensor()])
            dataset = THINGSBehavior(root=path_to_things, aligned=False, download=False, transform=transform, cc0=cc0, return_names=conditional or optim)

            loader = DataLoader(
                dataset=dataset,
                batch_size=1,
            )

            ckpt_name = model_params["ckpt"]
            if "stable-diffusion-3" in ckpt_name:
                encoder = SD3TransformerBlockEncoder(
                    checkpoint_name=ckpt_name,
                    noise_level=model_params["noise"],
                    extraction_module_name=module_type,
                )
            else:
                encoder = StableDiffusionEncoder(
                    checkpoint_name=ckpt_name,
                    noise_level=model_params["noise"],
                    extraction_module_name=module_type,
            )

            features = []
            with torch.no_grad():
                for sample in tqdm(loader):
                    img = (sample[0] if conditional or optim else sample).cuda()
                    if captions:
                        prompt = caption_dict[sample[1][0]]
                    elif optim:
                        prompt = caption_dict[sample[1][0]]
                    elif conditional:
                        prompt = ("a photo of a " + sample[1][0].split(".")[0].replace("_", " "))
                        prompt = ''.join([li for li in prompt if not li.isdigit()])
                    else:
                        prompt = None

                    avg_embedding = None
                    for i in range(n_to_avg):
                        seed = 42 + i * 1000
                        if eval_txt_enc:
                            try:
                                embedding = encoder.encode_prompt(prompt=prompt, batch_size=1, clip_padding=True)[0, eval_txt_enc_id, :]
                            except:
                                embedding = encoder.encode_prompt(prompt=prompt, clip_padding=True, seed=seed)[0, eval_txt_enc_id, :]
                        elif optim and not conditional:
                            embedding = torch.tensor(caption_dict[sample[1][0]].flatten())
                        else:
                            embedding = encoder(img, prompt=prompt, stop_early=True, seed=seed)
                            if pool:
                                # Spatial mean for 4D; token mean for 3D
                                if embedding.ndim == 4:
                                    embedding = torch.mean(embedding, dim=[-2, -1])
                                elif embedding.ndim == 3:
                                    token_axis = 1 if embedding.shape[1] >= embedding.shape[2] else 2
                                    embedding = torch.mean(embedding, dim=token_axis)

                        if avg_embedding is not None:
                            avg_embedding += embedding * (1 / n_to_avg)
                        else:
                            avg_embedding = embedding * (1 / n_to_avg)

                    features.append(
                        avg_embedding.detach().to(torch.float32).cpu()
                    )
            if features[0].shape[0] != 1:
                features = torch.stack(features).numpy()
            else:
                features = torch.cat(features).numpy()
        else:
            # thingsvision models
            # Get which module to query
            if path_to_model_dict is not None:
                with open(path_to_model_dict, "r") as f:
                    layer = json.load(f)[model_name][module_type]["module_name"]
            else:
                layer = module_type

            extractor = get_extractor(
                model_name=name,
                source=source,
                device="cuda" if torch.cuda.is_available() else "cpu",
                pretrained=pretrained,
                model_parameters=model_params,
            )
            dataset = THINGSBehavior(root=path_to_things,
                                     aligned=False,
                                     download=False,
                                     transform=extractor.get_transformations(),
                                     cc0=cc0)

            batches = DL(
                dataset=dataset,
                batch_size=128,
                backend=extractor.get_backend(),
            )

            with torch.no_grad():
                features = extractor.extract_features(
                    batches=batches,
                    module_name=layer,
                    flatten_acts=not pool,
                )
                if pool:
                    features = np.mean(np.mean(features, axis=-1), axis=-1)

        if not pretrained:
            model_name += "-untrained"

        if save:
            print("Saving to", path_to_embeddings)
            save_pickle(data=features,
                        file_path=path_to_embeddings,
                        model_name=model_name,
                        source=source,
                        module_type=module_type)

    return features