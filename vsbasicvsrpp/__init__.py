from __future__ import annotations

import math
import os
import warnings

import numpy as np
import torch
import torch.nn.functional as F
import vapoursynth as vs

from . import hack_registry  # isort: split
from mmengine.registry import MODELS
from mmengine.runner import load_checkpoint

from .basicvsr import BasicVSR
from .basicvsr_plusplus_net import BasicVSRPlusPlusNet

__version__ = "2.1.0"

os.environ["CUDA_MODULE_LOADING"] = "LAZY"

warnings.filterwarnings("ignore", "The given NumPy array is not writable")

model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models")


@torch.inference_mode()
def basicvsrpp(
    clip: vs.VideoNode,
    device_index: int = 0,
    model: int = 1,
    length: int = 15,
    cpu_cache: bool = False,
    tile: list[int] = [0, 0],
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
    :param length:          Length of sequence to process.
    :param cpu_cache:       Send the intermediate features to CPU.
                            This saves GPU memory, but slows down the inference speed.
    :param tile:            Tile width and height. As too large images result in the out of GPU memory issue, so this
                            tile option will first crop input images into tiles, and then process each of them. Finally,
                            they will be merged into one image. 0 denotes for do not use tile.
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

    if not isinstance(tile, list) or len(tile) != 2:
        raise vs.Error("basicvsrpp: tile must be a list with 2 items")

    if os.path.getsize(os.path.join(model_dir, "basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f.pth")) == 0:
        raise vs.Error("basicvsrpp: model files have not been downloaded. run 'python -m vsbasicvsrpp' first")

    torch.set_float32_matmul_precision("high")

    dtype = torch.half if clip.format.bits_per_sample == 16 else torch.float

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
    module.eval().to(device, dtype)

    if is_low_res_input:
        min_res = 64
        modulo = 1
        scale = 4
    else:
        min_res = 256
        modulo = 4
        scale = 1

    if all(t > 0 for t in tile):
        pad_w = math.ceil(max(tile[0] + 2 * tile_pad, min_res) / modulo) * modulo
        pad_h = math.ceil(max(tile[1] + 2 * tile_pad, min_res) / modulo) * modulo
    else:
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

            if all(t > 0 for t in tile):
                output = tile_process(img, scale, tile, tile_pad, device, pad_w, pad_h, module)
            else:
                img = img.to(device, non_blocking=True).clamp(0.0, 1.0)

                h, w = img.shape[3:]
                if need_pad := pad_w - w > 0 or pad_h - h > 0:
                    img = F.pad(img, (0, pad_w - w, 0, pad_h - h, 0, 0), "replicate")

                output = module(img)
                if need_pad:
                    output = output[:, :, :, : h * scale, : w * scale]

            output = output.squeeze(0).detach().cpu().numpy()
            for i in range(output.shape[0]):
                cache[str(n + i)] = output[i]

        return ndarray_to_frame(cache[str(n)], f[1].copy())

    new_clip = clip.std.BlankClip(width=clip.width * scale, height=clip.height * scale, keep=True)
    new_clip = new_clip.std.CopyFrameProps(clip)
    return new_clip.std.ModifyFrame([clip, new_clip], inference)


def frame_to_tensor(frame: vs.VideoFrame) -> torch.Tensor:
    return torch.stack([torch.from_numpy(np.asarray(frame[plane])) for plane in range(frame.format.num_planes)])


def ndarray_to_frame(array: np.ndarray, frame: vs.VideoFrame) -> vs.VideoFrame:
    for plane in range(frame.format.num_planes):
        np.copyto(np.asarray(frame[plane]), array[plane])
    return frame


def tile_process(
    img: torch.Tensor,
    scale: int,
    tile: list[int],
    tile_pad: int,
    device: torch.device,
    pad_w: int,
    pad_h: int,
    module: torch.nn.Module,
) -> torch.Tensor:
    batch, length, channel, height, width = img.shape
    output_shape = (batch, length, channel, height * scale, width * scale)

    # start with black image
    output = img.new_zeros(output_shape)

    tiles_x = math.ceil(width / tile[0])
    tiles_y = math.ceil(height / tile[1])

    # loop over all tiles
    for y in range(tiles_y):
        for x in range(tiles_x):
            # extract tile from input image
            ofs_x = x * tile[0]
            ofs_y = y * tile[1]

            # input tile area on total image
            input_start_x = ofs_x
            input_end_x = min(ofs_x + tile[0], width)
            input_start_y = ofs_y
            input_end_y = min(ofs_y + tile[1], height)

            # input tile area on total image with padding
            input_start_x_pad = max(input_start_x - tile_pad, 0)
            input_end_x_pad = min(input_end_x + tile_pad, width)
            input_start_y_pad = max(input_start_y - tile_pad, 0)
            input_end_y_pad = min(input_end_y + tile_pad, height)

            # input tile dimensions
            input_tile_width = input_end_x - input_start_x
            input_tile_height = input_end_y - input_start_y

            input_tile = img[:, :, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]
            input_tile = input_tile.to(device, non_blocking=True).clamp(0.0, 1.0)

            h, w = input_tile.shape[3:]
            if need_pad := pad_w - w > 0 or pad_h - h > 0:
                input_tile = F.pad(input_tile, (0, pad_w - w, 0, pad_h - h, 0, 0), "replicate")

            # process tile
            output_tile = module(input_tile)
            if need_pad:
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
