import torch
import torch.nn as nn
import torch.nn.functional as F

from safetensors.torch import load_model

import comfy.utils
import folder_paths
import latent_preview

class TinySanaDecoder(nn.Sequential):
    leaky = nn.LeakyReLU(0.2)

    @staticmethod
    def conv(c_in, c_out, k_s):
        return nn.Conv2d(c_in, c_out, kernel_size=k_s, padding="same")

    class Block(nn.Module):
        def __init__(self, c_in, c_out, factor):
            super().__init__()
            f_sq = factor * factor
            self.conv_a = TinySanaDecoder.conv(c_in, c_in, 3)
            self.conv_b = TinySanaDecoder.conv(c_in * 2, c_in, 3)
            self.conv_c = TinySanaDecoder.conv(c_in * 3, c_out * f_sq, 3)
            self.conv_l = TinySanaDecoder.conv(c_out * f_sq, c_out * f_sq, 1)
            self.shuffle = nn.PixelShuffle(factor)

        def forward(self, x):
            a = TinySanaDecoder.leaky(self.conv_a(x))
            ax = torch.cat((a, x), dim=1)
            b = TinySanaDecoder.leaky(self.conv_b(ax))
            bax = torch.cat((b, ax), dim=1)
            c = TinySanaDecoder.leaky(self.conv_c(bax))
            x = TinySanaDecoder.leaky(self.conv_l(c))
            x = self.shuffle(x)
            return x

    def __init__(self):
        super().__init__(
            TinySanaDecoder.Block(32, 128, 1),
            TinySanaDecoder.Block(128, 128, 1),
            TinySanaDecoder.Block(128, 128, 1),
            TinySanaDecoder.Block(128, 128, 2),
            TinySanaDecoder.Block(128, 128, 1),
            TinySanaDecoder.Block(128, 128, 1),
            TinySanaDecoder.Block(128, 128, 1),
            TinySanaDecoder.Block(128, 64, 2),
            TinySanaDecoder.Block(64, 64, 1),
            TinySanaDecoder.Block(64, 64, 1),
            TinySanaDecoder.Block(64, 64, 1),
            TinySanaDecoder.Block(64, 32, 2),
            TinySanaDecoder.Block(32, 32, 1),
            TinySanaDecoder.conv(32, 3, 3),
        )

class TinySanaPreviewer(latent_preview.LatentPreviewer):
    scale_factor = 0.41407

    def __init__(self, tiny, dtype):
        self.tiny = tiny
        self.dtype = dtype

    def decode_latent_to_preview(self, x0):
        x_sample = self.tiny((x0[:1] / TinySanaPreviewer.scale_factor).to(dtype=self.dtype))[0].movedim(0, 2)
        return latent_preview.preview_to_image(x_sample)

prepare_callback_original = latent_preview.prepare_callback

def prepare_callback_patch(model, steps, x0_output_dict=None):
    global prepare_callback_original

    if not model.attachments.get("tsd", False):
        return prepare_callback_original(model, steps, x0_output_dict)

    node = model.attachments.get("tsd_node")

    tsd = None
    tsd_name = model.attachments.get("tsd_name", "tsd.safetensors")

    if node.tsd_name is None or node.tsd is None or node.tsd_name != tsd_name:
        path = folder_paths.get_full_path_or_raise("vae_approx", model.attachments.get("tsd_name", "tsd.safetensors"))
        tsd = TinySanaDecoder()
        load_model(tsd, path)
        tsd.requires_grad_(False)
        node.tsd = tsd
        node.tsd_name = tsd_name
    else:
        tsd = node.tsd

    dtype = torch.bfloat16
    if model.attachments["tsd_dtype"] == "float32":
        dtype = torch.float32
    elif model.attachments["tsd_dtype"] == "float16":
        dtype = torch.float16
    tsd.to(device=model.load_device, dtype=dtype)

    previewer = TinySanaPreviewer(tsd, dtype)

    pbar = comfy.utils.ProgressBar(steps)

    def callback(step, x0, x, total_steps):
        if x0_output_dict is not None:
            x0_output_dict["x0"] = x0
        preview_bytes = None
        if previewer:
            preview_bytes = previewer.decode_latent_to_preview_image("JPEG", x0)
        pbar.update_absolute(step + 1, total_steps, preview_bytes)

    return callback

latent_preview.prepare_callback = prepare_callback_patch

class TinySanaPreview:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "previewer_model": (folder_paths.get_filename_list("vae_approx"), ),
                "dtype": (("bf16", "fp32", "fp16"), {"default": "bf16"}),
            }
        }
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("patched_model",)
    OUTPUT_TOOLTIPS = ("The patched model.",)
    FUNCTION = "patch"
    CATEGORY = "latent"
    DESCRIPTION = "Real-time previews for Sana models."

    def __init__(self):
        super().__init__()
        self.tsd = None
        self.tsd_name = None

    def patch(self, model, previewer_model, dtype):
        m = model.clone()
        m.attachments["tsd"] = True
        m.attachments["tsd_name"] = previewer_model
        m.attachments["tsd_dtype"] = dtype
        m.attachments["tsd_node"] = self
        return (m,)

NODE_CLASS_MAPPINGS = {
    "TinySanaPreview": TinySanaPreview,
}
