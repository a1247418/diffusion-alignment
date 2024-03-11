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
