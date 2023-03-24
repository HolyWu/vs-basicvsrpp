import os

import requests
from tqdm import tqdm


def download_model(url: str) -> None:
    filename = url.split("/")[-1]
    r = requests.get(url, stream=True)
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "models", filename), "wb") as f:
        with tqdm(
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            miniters=1,
            desc=filename,
            total=int(r.headers.get("content-length", 0)),
        ) as pbar:
            for chunk in r.iter_content(chunk_size=4096):
                f.write(chunk)
                pbar.update(len(chunk))


if __name__ == "__main__":
    url = "https://github.com/HolyWu/vs-basicvsrpp/releases/download/model/"
    models = [
        "basicvsr_plusplus_c64n7_8x1_300k_vimeo90k_bd_20210305-ab315ab1",
        "basicvsr_plusplus_c64n7_8x1_300k_vimeo90k_bi_20210305-4ef437e2",
        "basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f",
        "basicvsr_plusplus_c128n25_ntire_decompress_track1_20210223-7b2eba02",
        "basicvsr_plusplus_c128n25_ntire_decompress_track2_20210314-eeae05e6",
        "basicvsr_plusplus_c128n25_ntire_decompress_track3_20210304-6daf4a40",
        "basicvsr_plusplus_c128n25_ntire_vsr_20210311-1ff35292",
        "basicvsr_plusplus_deblur_dvd-ecd08b7f",
        "basicvsr_plusplus_deblur_gopro-3c5bb9b5",
        "basicvsr_plusplus_denoise-28f6920c",
        "spynet_20210409-c6c1bd09",
    ]
    for model in models:
        download_model(url + model + ".pth")
