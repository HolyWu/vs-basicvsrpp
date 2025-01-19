# BasicVSR++
Improving Video Super-Resolution with Enhanced Propagation and Alignment, based on https://github.com/ckkelvinchan/BasicVSR_PlusPlus.


## Dependencies
- [PyTorch](https://pytorch.org/get-started/) 2.5.1 or later
- [VapourSynth](http://www.vapoursynth.com/) R66 or later


## Installation
```
pip install -U vsbasicvsrpp
```

If you want to download all models at once, run `python -m vsbasicvsrpp`. If you prefer to only download the model you
specified at first run, set `auto_download=True` in `basicvsrpp()`.


## Usage
```python
from vsbasicvsrpp import basicvsrpp

ret = basicvsrpp(clip)
```

See `__init__.py` for the description of the parameters.
