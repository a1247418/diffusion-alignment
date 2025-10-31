import torch
import numpy as np
from typing import Optional, Union

from matplotlib import pyplot as plt
from diffusers import (
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionImg2ImgPipeline,
    AutoPipelineForImage2Image,
)


class Stop(Exception):
    def __init__(self, message, errors):
        # Call the base class constructor with the parameters it needs
        super().__init__(message)


class StableDiffusionEncoder(torch.nn.Module):
    def __init__(
        self,
        checkpoint_name: str,
        noise_level: int,
        extraction_module_name: str = "down_blocks.3.resnets.1",
        guidance_scale: int = 7,
        device: str = "cuda",
    ):
        super(StableDiffusionEncoder, self).__init__()

        self.device = device
        self.guidance_scale = guidance_scale
        self.generator = torch.Generator(device=device)
        self.noise_level = noise_level / 100.0

        if checkpoint_name == "stabilityai/stable-diffusion-2-1":
            self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                checkpoint_name
            ).to(device)
        elif checkpoint_name == "stabilityai/sd-turbo":
            self.pipe = AutoPipelineForImage2Image.from_pretrained(
                checkpoint_name, torch_dtype=torch.float16, variant="fp16"
            ).to("cuda")
        else:
            print("Unknown checkpoint, defaulting to StableDiffusionImg2ImgPipeline.")
            self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                checkpoint_name
            ).to(device)

        try:
            self.extraction_module = [
                m
                for n, m in self.pipe.unet.named_modules()
                if n == extraction_module_name
            ][0]
        except IndexError:
            raise ValueError(
                f"Could not find extraction module {extraction_module_name}"
            )

    def forward(
        self,
        x: torch.Tensor,
        prompt: Optional[Union[str, torch.Tensor]] = None,
        plot_outputs: bool = False,
        num_inference_steps: int = 20,
        stop_early: bool = False,
        seed: int = 42,
    ):
        self.generator.manual_seed(seed)

        if 1/num_inference_steps > self.noise_level/100:
            # Setting the inference steps to 100 if the noise level is smaller than one step
            num_inference_steps = 100

        prompt_embeds = None
        if prompt is None:
            conditional = False
            prompt = ""
            gs = 0
        else:
            conditional = True
            gs = self.guidance_scale
            if type(prompt) != str and type(prompt) != np.str_:
                prompt_embeds = torch.tensor(prompt).to(self.device)
                prompt = None

        activations = []

        def get_activations():
            def hook(model, input, output):
                activations.append(output.detach()[int(conditional)])
                if stop_early and len(activations) == 1:
                    raise Stop("Stop early", None)

            return hook

        hook = self.extraction_module.register_forward_hook(get_activations())
        with torch.no_grad():
            try:
                images = self.pipe(
                    prompt=prompt,
                    prompt_embeds=prompt_embeds,
                    image=x.to(self.device),
                    strength=self.noise_level,
                    guidance_scale=gs,
                    num_inference_steps=num_inference_steps,
                    generator=self.generator,
                ).images
            except Stop:
                images = []
                pass
        hook.remove()

        embedding = activations[0]

        if plot_outputs:
            for img in images:
                plt.imshow(img)
                plt.show()

        return embedding

    def encode_prompt(self, prompt: str, clip_padding: bool = False, seed: int = 42):
        self.generator.manual_seed(seed)
        prompt_embeds, _ = self.pipe.encode_prompt(
            prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
        )

        if clip_padding:
            # clip padding tokens
            attention_mask = self.pipe.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.pipe.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).attention_mask
            prompt_embeds = prompt_embeds[:, attention_mask[0] != 0]

        return prompt_embeds


# --- SD3/SD3.5 support ---

def _select_tensor_from_output(out):
    """
    Prefer hidden states. Fall back to the largest tensor by numel.
    Handles SD3 MMDiT blocks that may return tensors, tuples/lists, or dicts.
    """
    candidates = []

    def consider(t):
        if isinstance(t, torch.Tensor) and t.ndim >= 2 and t.numel() > 0:
            candidates.append(t)

    if isinstance(out, torch.Tensor):
        consider(out)
    elif isinstance(out, dict):
        for k in ("hidden_states", "last_hidden_state", "sample", "x"):
            v = out.get(k, None)
            if isinstance(v, torch.Tensor):
                consider(v)
        for v in out.values():
            if isinstance(v, torch.Tensor):
                consider(v)
            elif isinstance(v, (list, tuple)):
                for u in v:
                    if isinstance(u, torch.Tensor):
                        consider(u)
    elif isinstance(out, (list, tuple)):
        for v in out:
            if isinstance(v, torch.Tensor):
                consider(v)
            elif isinstance(v, (list, tuple)):
                for u in v:
                    if isinstance(u, torch.Tensor):
                        consider(u)

    if not candidates:
        return None
    return max(candidates, key=lambda t: t.numel())


class SD3TransformerBlockEncoder(torch.nn.Module):
    def __init__(
        self,
        checkpoint_name: str,
        noise_level: int,
        extraction_module_name: str = "transformer_blocks.8",
        guidance_scale: float = 3.5,
        device: str = "cuda",
    ):
        super(SD3TransformerBlockEncoder, self).__init__()

        self.device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.guidance_scale = guidance_scale
        self.generator = torch.Generator(device=self.device)
        self.noise_level = noise_level / 100.0

        # dtype policy: bf16 if supported else fp16 on CUDA; fp32 on CPU
        if self.device == "cuda":
            preferred_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            preferred_dtype = torch.float32

        self.pipe = AutoPipelineForImage2Image.from_pretrained(
            checkpoint_name,
            torch_dtype=preferred_dtype,
        ).to(self.device)

        try:
            self.pipe.set_progress_bar_config(disable=True)
        except Exception:
            pass

        # Resolve extraction module from transformer blocks
        transformer = getattr(self.pipe, "transformer", None)
        blocks = getattr(transformer, "transformer_blocks", None) if transformer is not None else None

        idx = None
        if isinstance(extraction_module_name, str) and extraction_module_name.startswith("transformer_blocks."):
            try:
                idx = int(extraction_module_name.split(".")[-1])
            except Exception:
                idx = None

        if blocks is not None and isinstance(blocks, (list, tuple)) and len(blocks) > 0 and idx is not None:
            if idx < 0:
                idx = 0
            if idx >= len(blocks):
                print(f"Requested block index {idx} out of range; clamping to {len(blocks)-1}.")
                idx = len(blocks) - 1
            self.extraction_module = blocks[idx]
            print(f"SD3TransformerBlockEncoder: using module 'transformer_blocks.{idx}'")
        else:
            # Fallback: try named module lookup, else last module
            try:
                self.extraction_module = dict(transformer.named_modules())[extraction_module_name]
                print(f"SD3TransformerBlockEncoder: using named module '{extraction_module_name}'")
            except Exception:
                all_mods = list(transformer.modules()) if transformer is not None else []
                self.extraction_module = all_mods[-1] if len(all_mods) > 0 else transformer
                print("SD3TransformerBlockEncoder: falling back to last transformer module")

    def forward(
        self,
        x: torch.Tensor,
        prompt: Optional[Union[str, torch.Tensor]] = None,
        plot_outputs: bool = False,
        num_inference_steps: int = 100,
        stop_early: bool = False,
        seed: int = 42,
    ):
        self.generator.manual_seed(seed)

        prompt_embeds = None
        if prompt is None or (isinstance(prompt, str) and len(prompt) == 0):
            cfg = 0.0
            prompt_to_use = ""
        else:
            cfg = float(self.guidance_scale)
            if isinstance(prompt, (np.ndarray, torch.Tensor)):
                prompt_embeds = torch.as_tensor(prompt).to(self.device)
                prompt_to_use = None
            else:
                prompt_to_use = str(prompt)

        captured = []

        def hook_fn(_m, _inp, out):
            sel = _select_tensor_from_output(out)
            if sel is not None:
                captured.append(sel.detach())
                if stop_early:
                    raise Stop("Stop early", None)

        handle = self.extraction_module.register_forward_hook(lambda m, i, o: hook_fn(m, i, o))
        with torch.no_grad():
            try:
                _ = self.pipe(
                    prompt=prompt_to_use,
                    prompt_embeds=prompt_embeds,
                    image=x.to(self.device),
                    strength=self.noise_level,
                    guidance_scale=cfg,
                    num_inference_steps=num_inference_steps,
                    generator=self.generator,
                )
            except Stop:
                pass
        handle.remove()

        if not captured:
            raise RuntimeError("No activation captured at the chosen noise level.")

        embedding = captured[0].to(torch.float32)

        if plot_outputs:
            pass

        return embedding
