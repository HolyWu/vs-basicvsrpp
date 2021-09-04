# BasicVSR++
BasicVSR++: Improving Video Super-Resolution with Enhanced Propagation and Alignment

Ported from https://github.com/open-mmlab/mmediting


## Dependencies
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive), required by `mmcv-full` to compile CUDA ops. Install the same version as in `PyTorch`.
- [mmcv-full](https://github.com/open-mmlab/mmcv#installation)
- [NumPy](https://numpy.org/install)
- [PyTorch](https://pytorch.org/get-started), preferably with CUDA. Note that `torchaudio` is not required and hence can be omitted from the command.
- [VapourSynth](http://www.vapoursynth.com/)


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
