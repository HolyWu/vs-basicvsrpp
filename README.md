# BasicVSR++
Improving Video Super-Resolution with Enhanced Propagation and Alignment, based on https://github.com/ckkelvinchan/BasicVSR_PlusPlus.


## Dependencies
- [mmcv](https://github.com/open-mmlab/mmcv#installation) >=2.0.0
- [NumPy](https://numpy.org/install)
- [PyTorch](https://pytorch.org/get-started) >=1.13.1
- [VapourSynth](http://www.vapoursynth.com/) R55+


## Installation
```
pip install -U openmim
mim install "mmcv>=2.0.0"

pip install -U vsbasicvsrpp
python -m vsbasicvsrpp
```


## Usage
```python
from vsbasicvsrpp import basicvsrpp

ret = basicvsrpp(clip)
```

See `__init__.py` for the description of the parameters.
