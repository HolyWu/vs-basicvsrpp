from __future__ import annotations

import math
import os

import numpy as np
import torch
import torch.nn.functional as F
import vapoursynth as vs
from mmengine.runner import load_checkpoint

from .basicvsr import BasicVSR
from .basicvsr_plusplus_net import BasicVSRPlusPlusNet
from .registry import MODELS

__version__ = "2.1.0"

os.environ["CUDA_MODULE_LOADING"] = "LAZY"

model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models")


@torch.inference_mode()
def basicvsrpp(
    clip: vs.VideoNode,
    device_index: int | None = None,
    model: int = 1,
    length: int = 15,
    cpu_cache: bool = False,
    tile_w: int = 0,
    tile_h: int = 0,
    tile_pad: int = 16,
) -> vs.VideoNode:
    """Improving Video Super-Resolution with Enhanced Propagation and Alignment

    :param clip:            Clip to process. Only RGBH and RGBS formats are supported.
                            RGBH performs inference in FP16 mode while RGBS performs inference in FP32 mode.
    :param device_index:    Device ordinal of the GPU.
    :param model:           Model to use.
                            0 = Video Super-Resolution (REDS)
                            1 = Video Super-Resolution (Vimeo-90K BI degradation)
                            2 = Video Super-Resolution (Vimeo-90K BD degradation)
                            3 = NTIRE 2021 Video Super-Resolution
                            4 = NTIRE 2021 Quality Enhancement of Compressed Video - Track 1
                            5 = NTIRE 2021 Quality Enhancement of Compressed Video - Track 2
                            6 = NTIRE 2021 Quality Enhancement of Compressed Video - Track 3
                            7 = Video Deblurring (DVD)
                            8 = Video Deblurring (GoPro)
                            9 = Video Denoising
    :param length:          Sequence length that the model processes.
    :param cpu_cache:       Send the intermediate features to CPU.
                            This saves GPU memory, but slows down the inference speed.
    :param tile_w:          Tile width. As too large images result in the out of GPU memory issue, so this tile option
                            will first crop input images into tiles, and then process each of them. Finally, they will
                            be merged into one image. 0 denotes for do not use tile.
    :param tile_h:          Tile height.
    :param tile_pad:        Pad size for each tile, to remove border artifacts.
    """
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error("basicvsrpp: this is not a clip")

    if clip.format.id not in [vs.RGBH, vs.RGBS]:
        raise vs.Error("basicvsrpp: only RGBH and RGBS formats are supported")

    if not torch.cuda.is_available():
        raise vs.Error("basicvsrpp: CUDA is not available")

    if model not in range(10):
        raise vs.Error("basicvsrpp: model must be 0, 1, 2, 3, 4, 5, 6, 7, 8, or 9")

    if length < 1:
        raise vs.Error("basicvsrpp: length must be at least 1")

    if os.path.getsize(os.path.join(model_dir, "basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f.pth")) == 0:
        raise vs.Error("basicvsrpp: model files have not been downloaded. run 'python -m vsbasicvsrpp' first")

    torch.set_float32_matmul_precision("high")

    fp16 = clip.format.bits_per_sample == 16

    device = torch.device("cuda", device_index)

    match model:
        case 0:
            model_name = "basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f.pth"
            mid_channels = 64
            num_blocks = 7
            is_low_res_input = True
        case 1:
            model_name = "basicvsr_plusplus_c64n7_8x1_300k_vimeo90k_bi_20210305-4ef437e2.pth"
            mid_channels = 64
            num_blocks = 7
            is_low_res_input = True
        case 2:
            model_name = "basicvsr_plusplus_c64n7_8x1_300k_vimeo90k_bd_20210305-ab315ab1.pth"
            mid_channels = 64
            num_blocks = 7
            is_low_res_input = True
        case 3:
            model_name = "basicvsr_plusplus_c128n25_ntire_vsr_20210311-1ff35292.pth"
            mid_channels = 128
            num_blocks = 25
            is_low_res_input = True
        case 4:
            model_name = "basicvsr_plusplus_c128n25_ntire_decompress_track1_20210223-7b2eba02.pth"
            mid_channels = 128
            num_blocks = 25
            is_low_res_input = False
        case 5:
            model_name = "basicvsr_plusplus_c128n25_ntire_decompress_track2_20210314-eeae05e6.pth"
            mid_channels = 128
            num_blocks = 25
            is_low_res_input = False
        case 6:
            model_name = "basicvsr_plusplus_c128n25_ntire_decompress_track3_20210304-6daf4a40.pth"
            mid_channels = 128
            num_blocks = 25
            is_low_res_input = False
        case 7:
            model_name = "basicvsr_plusplus_deblur_dvd-ecd08b7f.pth"
            mid_channels = 64
            num_blocks = 15
            is_low_res_input = False
        case 8:
            model_name = "basicvsr_plusplus_deblur_gopro-3c5bb9b5.pth"
            mid_channels = 64
            num_blocks = 15
            is_low_res_input = False
        case 9:
            model_name = "basicvsr_plusplus_denoise-28f6920c.pth"
            mid_channels = 64
            num_blocks = 15
            is_low_res_input = False

    if is_low_res_input:
        min_res = 64
        modulo = 1
        scale = 4
    else:
        min_res = 256
        modulo = 4
        scale = 1

    model_path = os.path.join(model_dir, model_name)
    spynet_path = os.path.join(model_dir, "spynet_20210409-c6c1bd09.pth")

    cfg = dict(
        type="BasicVSR",
        generator=dict(
            type="BasicVSRPlusPlusNet",
            mid_channels=mid_channels,
            num_blocks=num_blocks,
            is_low_res_input=is_low_res_input,
            spynet_pretrained=spynet_path,
            cpu_cache=cpu_cache,
        ),
    )

    module = MODELS.build(cfg)
    load_checkpoint(module, model_path, map_location="cpu", logger="silent")
    module.eval().to(device)
    if fp16:
        module.half()

    pad_w = math.ceil(max(clip.width, min_res) / modulo) * modulo
    pad_h = math.ceil(max(clip.height, min_res) / modulo) * modulo

    cache = {}

    @torch.inference_mode()
    def inference(n: int, f: list[vs.VideoFrame]) -> vs.VideoFrame:
        if str(n) not in cache:
            cache.clear()

            img = [frame_to_tensor(f[0])]
            for i in range(1, length):
                if n + i >= clip.num_frames:
                    break
                img.append(frame_to_tensor(clip.get_frame(n + i)))

            img = torch.stack(img).unsqueeze(0)

            if tile_w > 0 and tile_h > 0:
                output = tile_process(img, scale, tile_w, tile_h, tile_pad, device, min_res, modulo, module)
            else:
                img = img.to(device).clamp(0.0, 1.0)

                h, w = img.shape[3:]
                mode = "reflect" if pad_w - w < w and pad_h - h < h else "replicate"
                img = F.pad(img, (0, pad_w - w, 0, pad_h - h, 0, 0), mode)

                output = module(img)
                output = output[:, :, :, : h * scale, : w * scale]

            output = output.squeeze(0).detach().cpu().numpy()
            for i in range(output.shape[0]):
                cache[str(n + i)] = output[i, :, :, :]

        return ndarray_to_frame(cache[str(n)], f[1].copy())

    new_clip = clip.std.BlankClip(width=clip.width * scale, height=clip.height * scale, keep=True)
    return new_clip.std.ModifyFrame([clip, new_clip], inference)


def frame_to_tensor(frame: vs.VideoFrame) -> torch.Tensor:
    array = np.stack([np.asarray(frame[plane]) for plane in range(frame.format.num_planes)])
    return torch.from_numpy(array)


def ndarray_to_frame(array: np.ndarray, frame: vs.VideoFrame) -> vs.VideoFrame:
    for plane in range(frame.format.num_planes):
        np.copyto(np.asarray(frame[plane]), array[plane, :, :])
    return frame


def tile_process(
    img: torch.Tensor,
    scale: int,
    tile_w: int,
    tile_h: int,
    tile_pad: int,
    device: torch.device,
    min_res: int,
    modulo: int,
    module: torch.nn.Module,
) -> torch.Tensor:
    batch, length, channel, height, width = img.shape
    output_shape = (batch, length, channel, height * scale, width * scale)

    # start with black image
    output = img.new_zeros(output_shape)

    tiles_x = math.ceil(width / tile_w)
    tiles_y = math.ceil(height / tile_h)

    # loop over all tiles
    for y in range(tiles_y):
        for x in range(tiles_x):
            # extract tile from input image
            ofs_x = x * tile_w
            ofs_y = y * tile_h

            # input tile area on total image
            input_start_x = ofs_x
            input_end_x = min(ofs_x + tile_w, width)
            input_start_y = ofs_y
            input_end_y = min(ofs_y + tile_h, height)

            # input tile area on total image with padding
            input_start_x_pad = max(input_start_x - tile_pad, 0)
            input_end_x_pad = min(input_end_x + tile_pad, width)
            input_start_y_pad = max(input_start_y - tile_pad, 0)
            input_end_y_pad = min(input_end_y + tile_pad, height)

            # input tile dimensions
            input_tile_width = input_end_x - input_start_x
            input_tile_height = input_end_y - input_start_y

            input_tile = img[:, :, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]
            input_tile = input_tile.to(device).clamp(0.0, 1.0)

            h, w = input_tile.shape[3:]
            pad_w = math.ceil(max(w, min_res) / modulo) * modulo
            pad_h = math.ceil(max(h, min_res) / modulo) * modulo
            mode = "reflect" if pad_w - w < w and pad_h - h < h else "replicate"
            input_tile = F.pad(input_tile, (0, pad_w - w, 0, pad_h - h, 0, 0), mode)

            # process tile
            output_tile = module(input_tile)
            output_tile = output_tile[:, :, :, : h * scale, : w * scale]

            # output tile area on total image
            output_start_x = input_start_x * scale
            output_end_x = input_end_x * scale
            output_start_y = input_start_y * scale
            output_end_y = input_end_y * scale

            # output tile area without padding
            output_start_x_tile = (input_start_x - input_start_x_pad) * scale
            output_end_x_tile = output_start_x_tile + input_tile_width * scale
            output_start_y_tile = (input_start_y - input_start_y_pad) * scale
            output_end_y_tile = output_start_y_tile + input_tile_height * scale

            # put tile into output image
            output[:, :, :, output_start_y:output_end_y, output_start_x:output_end_x] = output_tile[
                :, :, :, output_start_y_tile:output_end_y_tile, output_start_x_tile:output_end_x_tile
            ]

    return output
