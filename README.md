# BasicVSR++
BasicVSR++: Improving Video Super-Resolution with Enhanced Propagation and Alignment

Ported from https://github.com/open-mmlab/mmediting


## Dependencies
- [mmcv-full](https://github.com/open-mmlab/mmcv#installation)
- [NumPy](https://numpy.org/install)
- [PyTorch](https://pytorch.org/get-started), preferably with CUDA. Note that `torchaudio` is not required and hence can be omitted from the command.
- [VapourSynth](http://www.vapoursynth.com/)

Installing `mmcv-full` on Windows is a bit complicated as it requires Visual Studio and other tools to compile CUDA ops.
So I have uploaded the built file compiled with CUDA 11.1 for Windows users and you can install it by executing the following command.
```
pip install https://github.com/HolyWu/vs-basicvsrpp/releases/download/v1.0.0/mmcv_full-1.3.12-cp39-cp39-win_amd64.whl
```


## Installation
```
pip install --upgrade vsbasicvsrpp
python -m vsbasicvsrpp
```


## Usage
```python
from vsbasicvsrpp import BasicVSRPP

ret = BasicVSRPP(clip)
```

See `__init__.py` for the description of the parameters.
