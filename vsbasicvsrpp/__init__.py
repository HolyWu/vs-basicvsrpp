import math
import mmcv
import numpy as np
import os
import torch
import vapoursynth as vs
from .basicvsr import BasicVSR
from .basicvsr_pp import BasicVSRPlusPlus
from .builder import build_model


def BasicVSRPP(clip: vs.VideoNode, model: int=1, interval: int=30, tile_x: int=0, tile_y: int=0, tile_pad: int=16,
               device_type: str='cuda', device_index: int=0, fp16: bool=False) -> vs.VideoNode:
    '''
    BasicVSR++: Improving Video Super-Resolution with Enhanced Propagation and Alignment

    Support either x4 upsampling (for model 0-2) or same size output (for model 3-5).
    For model 0-2, input resolution must be at least 64.
    For model 3-5, input resolution must be at least 256 and mod-4.

    Parameters:
        clip: Clip to process. Only planar format with float sample type of 32 bit depth is supported.

        model: Model to use.
            0 = REDS
            1 = Vimeo-90K (BI)
            2 = Vimeo-90K (BD)
            3 = NTIRE 2021 Quality enhancement of heavily compressed videos Challenge - Track 1
            4 = NTIRE 2021 Quality enhancement of heavily compressed videos Challenge - Track 2
            5 = NTIRE 2021 Quality enhancement of heavily compressed videos Challenge - Track 3

        interval: Interval size.

        tile_x, tile_y: Tile width and height respectively, 0 for no tiling.
            It's recommended that the input's width and height is divisible by the tile's width and height respectively.
            Set it to the maximum value that your GPU supports to reduce its impact on the output.

        tile_pad: Tile padding.

        device_type: Device type on which the tensor is allocated. Must be 'cuda' or 'cpu'.

        device_index: Device ordinal for the device type.

        fp16: fp16 mode for faster and more lightweight inference on cards with Tensor Cores.
    '''
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error('BasicVSR++: this is not a clip')

    if clip.format.id != vs.RGBS:
        raise vs.Error('BasicVSR++: only RGBS format is supported')

    if model not in [0, 1, 2, 3, 4, 5]:
        raise vs.Error('BasicVSR++: model must be 0, 1, 2, 3, 4, or 5')

    if interval < 1:
        raise vs.Error('BasicVSR++: interval must be at least 1')

    device_type = device_type.lower()

    if device_type not in ['cuda', 'cpu']:
        raise vs.Error("BasicVSR++: device_type must be 'cuda' or 'cpu'")

    if device_type == 'cuda' and not torch.cuda.is_available():
        raise vs.Error('BasicVSR++: CUDA is not available')

    device = torch.device(device_type, device_index)
    if device_type == 'cuda':
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    if model == 0:
        model_name = 'basicvsr_plusplus_reds4.pth'
    elif model == 1:
        model_name = 'basicvsr_plusplus_vimeo90k_bi.pth'
    elif model == 2:
        model_name = 'basicvsr_plusplus_vimeo90k_bd.pth'
    elif model == 3:
        model_name = 'basicvsr_plusplus_ntire_decompress_track1.pth'
    elif model == 4:
        model_name = 'basicvsr_plusplus_ntire_decompress_track2.pth'
    else:
        model_name = 'basicvsr_plusplus_ntire_decompress_track3.pth'
    model_path = os.path.join(os.path.dirname(__file__), model_name)

    if model < 3:
        config_name = 'config012.py'
        scale = 4
    else:
        config_name = 'config345.py'
        scale = 1

    config = mmcv.Config.fromfile(os.path.join(os.path.dirname(__file__), config_name))

    model = build_model(config.model)
    mmcv.runner.load_checkpoint(model, model_path, strict=True)
    model.to(device)
    model.eval()
    if fp16:
        model.half()

    cache = {}

    def basicvsrpp(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
        nonlocal cache

        if str(n) not in cache.keys():
            cache.clear()

            imgs = [frame_to_tensor(f[0])]
            for i in range(1, interval):
                if (n + i) >= clip.num_frames:
                    break
                imgs.append(frame_to_tensor(clip.get_frame(n + i)))

            imgs = torch.stack(imgs)
            imgs = imgs.unsqueeze(0).to(device)
            if fp16:
                imgs = imgs.half()

            with torch.no_grad():
                if tile_x > 0 and tile_y > 0:
                    output = tile_process(imgs, scale, tile_x, tile_y, tile_pad, model)
                else:
                    output = model(imgs)

            output = output.squeeze(0).detach().cpu().numpy()
            for i in range(output.shape[0]):
                cache[str(n + i)] = output[i, :, :, :]

            del imgs
            torch.cuda.empty_cache()

        return ndarray_to_frame(cache[str(n)], f[1])

    new_clip = clip.std.BlankClip(width=clip.width * scale, height=clip.height * scale)
    return new_clip.std.ModifyFrame(clips=[clip, new_clip], selector=basicvsrpp)


def frame_to_tensor(f: vs.VideoFrame) -> torch.Tensor:
    arr = np.stack([np.asarray(f.get_read_array(plane)) for plane in range(f.format.num_planes)])
    return torch.from_numpy(arr)


def ndarray_to_frame(arr: np.ndarray, f: vs.VideoFrame) -> vs.VideoFrame:
    fout = f.copy()
    for plane in range(fout.format.num_planes):
        np.copyto(np.asarray(fout.get_write_array(plane)), arr[plane, :, :])
    return fout


def tile_process(img: torch.Tensor, scale: int, tile_x: int, tile_y: int, tile_pad: int, model: BasicVSR) -> torch.Tensor:
    batch, num, channel, height, width = img.shape
    output_height = height * scale
    output_width = width * scale
    output_shape = (batch, num, channel, output_height, output_width)

    # start with black image
    output = img.new_zeros(output_shape)

    tiles_x = math.ceil(width / tile_x)
    tiles_y = math.ceil(height / tile_y)

    # loop over all tiles
    for y in range(tiles_y):
        for x in range(tiles_x):
            # extract tile from input image
            ofs_x = x * tile_x
            ofs_y = y * tile_y

            # input tile area on total image
            input_start_x = ofs_x
            input_end_x = min(ofs_x + tile_x, width)
            input_start_y = ofs_y
            input_end_y = min(ofs_y + tile_y, height)

            # input tile area on total image with padding
            input_start_x_pad = max(input_start_x - tile_pad, 0)
            input_end_x_pad = min(input_end_x + tile_pad, width)
            input_start_y_pad = max(input_start_y - tile_pad, 0)
            input_end_y_pad = min(input_end_y + tile_pad, height)

            # input tile dimensions
            input_tile_width = input_end_x - input_start_x
            input_tile_height = input_end_y - input_start_y

            input_tile = img[:, :, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

            # upscale tile
            output_tile = model(input_tile)

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
            output[:, :, :, output_start_y:output_end_y, output_start_x:output_end_x] = \
                output_tile[:, :, :, output_start_y_tile:output_end_y_tile, output_start_x_tile:output_end_x_tile]

    return output
