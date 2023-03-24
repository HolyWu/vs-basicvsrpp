# BasicVSR++
Improving Video Super-Resolution with Enhanced Propagation and Alignment, based on https://github.com/ckkelvinchan/BasicVSR_PlusPlus.


## Dependencies
- [mmcv-full](https://github.com/open-mmlab/mmcv#installation) 1.7
- [NumPy](https://numpy.org/install)
- [PyTorch](https://pytorch.org/get-started) 1.13
- [VapourSynth](http://www.vapoursynth.com/) R55+


## Installation
```
pip install -U openmim vsbasicvsrpp
mim install "mmcv-full>=1.7.0"
python -m vsbasicvsrpp
```


## Usage
```python
from vsbasicvsrpp import basicvsrpp

ret = basicvsrpp(clip)
```

See `__init__.py` for the description of the parameters.
