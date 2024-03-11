import torch
import inspect
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import Optional, List, Tuple, Dict, Any, Union

from matplotlib import pyplot as plt
from torchvision.transforms import ToTensor, Compose, Resize, ToPILImage
from diffusers import (
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionImg2ImgPipeline,
    AutoPipelineForImage2Image,
)
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils.torch_utils import randn_tensor


class StableDiffusionEncoder(torch.nn.Module):
    def __init__(
        self,
        checkpoint_name: str,
        noise_level: Union[int, List[int]],
        extraction_module_name: str = "down_blocks.3.resnets.1",
        guidance_scale: int = 7,
        dtype: torch.dtype = torch.float32,
        device: str = "cuda",
        num_inference_steps=20,
    ):
        """
        Args:
            checkpoint_name: name of the hugginface checkpoint to load
            noise_level: noise level to use for encoding in percent
            extraction_module_name: name of the module to extract activations from
            guidance_scale: guidance scale for classifier free guidance
            dtype: dtype to use for encoding
            device: device to use for encoding
        """
        super(StableDiffusionEncoder, self).__init__()
        self.dtype = dtype
        self.device = device
        self.guidance_scale = guidance_scale
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(42)

        pipe = StableDiffusionPipeline.from_pretrained(checkpoint_name)

        self.vae = pipe.vae.to(device)
        self.unet = pipe.unet.to(device)
        self.unet.eval()
        self.scheduler = pipe.scheduler
        self.text_encoder = pipe.text_encoder.to(device)
        self.text_tokenizer = pipe.tokenizer

        # todo: make customizable
        self.prepper = VaeImageProcessor(
            do_resize=True,
            vae_scale_factor=8,
            resample="lanczos",
            do_normalize=True,
            do_binarize=False,
            do_convert_rgb=False,  # ?
            do_convert_grayscale=False,
        )

        try:
            self.extraction_module = [
                m for n, m in self.unet.named_modules() if n == extraction_module_name
            ][0]
        except IndexError:
            raise ValueError(
                f"Could not find extraction module {extraction_module_name}"
            )

        # set eta, if the schelduler allows it
        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        self.extra_step_kwargs = {}
        if accepts_eta:
            self.extra_step_kwargs["eta"] = 0.0  # 0 = deterministic, 1 = stochastic
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            self.extra_step_kwargs["generator"] = self.generator

        # Set up time
        # todo: less hacky
        step_multiplier = num_inference_steps / 100
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        if not type(noise_level) == int:
            noise_level = [int(nl * step_multiplier) for nl in noise_level]
            self.timesteps = torch.stack(
                [timesteps[(num_inference_steps - t)] for t in noise_level]
            )
        else:
            noise_level = int(noise_level * step_multiplier)
            self.timesteps = timesteps[
                (num_inference_steps - noise_level) : (
                    num_inference_steps + 1 - noise_level
                )
            ]

    # From https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_img2img.py#L77
    def retrieve_latents(
        self, encoder_output: torch.Tensor, sample_mode: str = "sample"
    ):
        if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
            return encoder_output.latent_dist.sample(self.generator)
        elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
            return encoder_output.latent_dist.mode()
        elif hasattr(encoder_output, "latents"):
            return encoder_output.latents
        else:
            print(encoder_output)
            raise AttributeError("Could not access latents of provided encoder_output")

    def encode_image(self, image, latent_timestep):
        if image.shape[1] == 4:
            init_latents = image
        else:
            image_prep = self.prepper.preprocess(image)
            with torch.no_grad():
                init_latents = self.retrieve_latents(self.vae.encode(image_prep))
            init_latents = self.vae.config.scaling_factor * init_latents
            init_latents = torch.cat([init_latents], dim=0)

        shape = init_latents.shape
        noise = randn_tensor(
            shape,
            generator=self.generator,
            device=torch.device(self.device),
            dtype=self.dtype,
        )

        init_latents = self.scheduler.add_noise(
            init_latents, noise, latent_timestep
        )  # torch.tensor([0]))#latent_timestep)
        latents = init_latents
        return latents

    # Adapted from https://github.com/huggingface/diffusers/blob/bf40d7d82a732a35e0f5b907e59064ee080cde9f/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_img2img.py#L325
    def _encode_prompt(self, prompt, batch_size, max_length=77, clip_padding=False):
        text_input = self.text_tokenizer(
            [prompt] * batch_size,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )

        if (
            hasattr(self.text_encoder.config, "use_attention_mask")
            and self.text_encoder.config.use_attention_mask
        ):
            attention_mask = text_input.attention_mask.to(self.device)
        else:
            attention_mask = None

        with torch.no_grad():
            if True:  # prompt == "":
                prompt_embeds = self.text_encoder(
                    text_input.input_ids.to(self.device), attention_mask=attention_mask
                )
                prompt_embeds = prompt_embeds[0]
            else:
                prompt_embeds = self.text_encoder(
                    text_input.input_ids.to(self.device),
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                prompt_embeds = prompt_embeds[-1][-1]
                prompt_embeds = self.text_encoder.text_model.final_layer_norm(
                    prompt_embeds
                )

        prompt_embeds = prompt_embeds.to(dtype=self.dtype, device=self.device)
        # bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        # prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        # prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
        if clip_padding:
            # clip padding tokens
            prompt_embeds = prompt_embeds[:, text_input.attention_mask[0] != 0]
        return prompt_embeds

    def encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
        max_length: int = 77,
    ):
        if prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = 1

        if prompt_embeds is None:
            text_inputs = self.text_tokenizer(
                prompt,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids

            if (
                hasattr(self.text_encoder.config, "use_attention_mask")
                and self.text_encoder.config.use_attention_mask
            ):
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            if clip_skip is None:
                prompt_embeds = self.text_encoder(
                    text_input_ids.to(device), attention_mask=attention_mask
                )
                prompt_embeds = prompt_embeds[0]
            else:
                prompt_embeds = self.text_encoder(
                    text_input_ids.to(device),
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                # Access the `hidden_states` first, that contains a tuple of
                # all the hidden states from the encoder layers. Then index into
                # the tuple to access the hidden states from the desired layer.
                prompt_embeds = prompt_embeds[-1][-(clip_skip + 1)]
                # We also need to apply the final LayerNorm here to not mess with the
                # representations. The `last_hidden_states` that we typically use for
                # obtaining the final prompt representations passes through the LayerNorm
                # layer.
                prompt_embeds = self.text_encoder.text_model.final_layer_norm(
                    prompt_embeds
                )

        if self.text_encoder is not None:
            prompt_embeds_dtype = self.text_encoder.dtype
        elif self.unet is not None:
            prompt_embeds_dtype = self.unet.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            bs_embed * num_images_per_prompt, seq_len, -1
        )

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = prompt_embeds.shape[1]
            uncond_input = self.text_tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if (
                hasattr(self.text_encoder.config, "use_attention_mask")
                and self.text_encoder.config.use_attention_mask
            ):
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(
                dtype=prompt_embeds_dtype, device=device
            )

            negative_prompt_embeds = negative_prompt_embeds.repeat(
                1, num_images_per_prompt, 1
            )
            negative_prompt_embeds = negative_prompt_embeds.view(
                batch_size * num_images_per_prompt, seq_len, -1
            )

        return prompt_embeds, negative_prompt_embeds

    def forward(
        self, x: torch.Tensor, prompt: Optional[str] = None, plot_outputs: bool = False
    ):
        batch_size = x.shape[0]
        latent_timestep = self.timesteps[:1].repeat(batch_size)

        # Noise image
        latents = self.encode_image(x, latent_timestep)

        # Embed prompt if necessary
        do_classifier_free_guidance = prompt is not None
        if do_classifier_free_guidance:
            if type(prompt) == str or type(prompt) == np.str_:
                # prompt_embeds = self.encode_prompt(prompt, batch_size)
                # uncond_prompt_embeds = self.encode_prompt("", batch_size)
                prompt_embeds, uncond_prompt_embeds = self.encode_prompt(
                    prompt=prompt,
                    device=self.device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=True,
                )
                print(
                    "embedding sizes:", prompt_embeds.shape, uncond_prompt_embeds.shape
                )
                try:
                    print(prompt_embeds[0, 12, 12], uncond_prompt_embeds[0, 12, 12])
                except:
                    pass
            else:
                # Embedding is given [bs, seq, dim]
                if len(prompt.shape) == 2:
                    prompt = prompt[None]
                if type(prompt) != torch.Tensor:
                    prompt = torch.tensor(prompt, device=self.device, dtype=self.dtype)
                prompt_embeds = prompt.expand(batch_size, -1, -1)
                # uncond_prompt_embeds = self.encode_prompt("", batch_size, max_length=prompt_embeds.shape[1])
                # prompt_embeds = self.encode_prompt(prompt, batch_size)
                # uncond_prompt_embeds = self.encode_prompt("", batch_size)
                _, uncond_prompt_embeds = self.encode_prompt(
                    prompt="",
                    device=self.device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=True,
                    max_length=prompt_embeds.shape[1],
                )

                if uncond_prompt_embeds.shape[1] != prompt_embeds.shape[1]:
                    # making sure prompt the same size as the uncond_prompt_embeds, in case max_len was not enforced
                    # TODO: this may cause problems
                    prompt_embeds = prompt_embeds.expand(
                        -1, uncond_prompt_embeds.shape[1], -1
                    )
                    print(
                        "embedding sizes:",
                        prompt_embeds.shape,
                        uncond_prompt_embeds.shape,
                    )

            prompt_embeds = torch.cat([uncond_prompt_embeds, prompt_embeds])
        else:
            # prompt_embeds = self.encode_prompt("", batch_size)
            prompt_embeds, uncond_prompt_embeds = self.encode_prompt(
                prompt="",
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
            )

        # Prepare hook for activations
        activations = []

        def get_activations():
            def hook(model, input, output):
                out = output[0].detach().clone()
                if do_classifier_free_guidance:
                    # Throw out un-conditional part
                    out_uncond, out = out.chunk(2)
                    print("ag", out_uncond.shape, out.shape)
                    try:
                        print("ag", out_uncond[70, 3, 3], out[70, 3, 3])
                    except:
                        pass
                else:
                    print("ag", out.shape)
                    try:
                        print("ag", out[70, 3, 3])
                    except:
                        pass
                activations.append(out)

            return hook

        hook = self.extraction_module.register_forward_hook(get_activations())

        # Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(
                batch_size
            )
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=self.device, dtype=latents.dtype)

        # Encode
        with torch.no_grad():
            for i, t in enumerate(
                self.timesteps
            ):  # This is only one timestep at the current version
                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

                # predict the noise residual
                try:
                    print("pe", prompt_embeds[:, 12, 12])
                except:
                    pass
                print("shapes", latent_model_input.shape, t.shape, prompt_embeds.shape)
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=None,
                    added_cond_kwargs=None,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                # compute the previous noisy sample x_t -> x_t-1
                if plot_outputs:
                    old_latents = latents.detach().clone()

                latents_all = self.scheduler.step(
                    noise_pred, t, latents, **self.extra_step_kwargs, return_dict=True
                )
                latents = latents_all.prev_sample
                latents_x0 = latents_all.pred_original_sample

                if plot_outputs:
                    # For validation purposes
                    image = x[0]

                    # Plotting
                    fig, axes = plt.subplots(
                        1, 4, figsize=(17, 5)
                    )  # Adjust the figure size as needed

                    axes[0].imshow(ToPILImage()(image.detach().cpu()))
                    axes[0].set_title("Original")
                    axes[0].axis("off")

                    axes[1].imshow(
                        self.prepper.postprocess(
                            self.vae.decode(
                                1 / self.vae.config.scaling_factor * old_latents
                            )
                            .sample.detach()
                            .cpu(),
                            output_type="pil",
                        )[0]
                    )
                    axes[1].set_title("$x_{t}$ (t=%d)" % t)
                    axes[1].axis("off")

                    axes[2].imshow(
                        self.prepper.postprocess(
                            self.vae.decode(
                                1 / self.vae.config.scaling_factor * latents
                            )
                            .sample.detach()
                            .cpu(),
                            output_type="pil",
                        )[0]
                    )
                    axes[2].set_title("Predicted $x_{t-1}$ (t=%d)" % t)
                    axes[2].axis("off")

                    axes[3].imshow(
                        self.prepper.postprocess(
                            self.vae.decode(
                                1 / self.vae.config.scaling_factor * latents_x0
                            )
                            .sample.detach()
                            .cpu(),
                            output_type="pil",
                        )[0]
                    )
                    axes[3].set_title(f"Predicted $x_{0}$ (t={t})")
                    axes[3].axis("off")

                    plt.show()

            hook.remove()
        try:
            # Reset Euler timestep counter
            self.scheduler._step_index = None
        except:
            pass
        return torch.cat(activations)


class Stop(Exception):
    def __init__(self, message, errors):
        # Call the base class constructor with the parameters it needs
        super().__init__(message)


class StableDiffusionEncoder2(torch.nn.Module):
    def __init__(
        self,
        checkpoint_name: str,
        noise_level: int,
        extraction_module_name: str = "down_blocks.3.resnets.1",
        guidance_scale: int = 7,
        device: str = "cuda",
    ):
        super(StableDiffusionEncoder2, self).__init__()

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
