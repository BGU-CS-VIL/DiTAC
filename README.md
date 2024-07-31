# Trainable Highly-expressive Activation Functions [ECCV 2024]

[Irit Chelly*](https://irita42.wixsite.com/mysite), [Shahaf E. Finder*](https://shahaffind.github.io/), [Shira Ifergane](https://www.linkedin.com/in/shira-ifergane/), and [Oren Freifeld](https://www.cs.bgu.ac.il/~orenfr/).

[![arXiv](https://img.shields.io/badge/arXiv-2407.07564-b31b1b.svg?style=flat)](https://arxiv.org/abs/2407.07564)

### Requirements
- python 3.12
- torch 2.2.2 
- difw 0.0.29
- scikit-learn 1.5.1

`difw` can be fragile, make sure that it compiles successfully. We've successfully compiled it with cuda version 12.2. 

Note that you might have to update c++ compiler and/or paths, e.g.:
```sh
conda install cxx-compiler -c conda-forge
export CPATH=/usr/local/cuda-12/targets/x86_64-linux/include:$CPATH
export LD_LIBRARY_PATH=/usr/local/cuda-12/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-12/bin:$PATH
```

### How to use
You can import the activation layer and use it within your network.
```python
import torch
from ditac import DiTAC

act = DiTAC()
x = torch.randn(8,32,20,20).to('cuda')

act(x)
```

You can recreate Fig. 1 using the sample code:
```sh
python run_regression.py
```

**Other experiments' code will be uploaded soon**

## License
This project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.


## Citation
If you find this repository helpful, please consider citing:
```
@inproceedings{chelly2024ditac,
  title     = {Trainable Highly-expressive Activation Functions},
  author    = {Chelly, Irit and Finder, Shahaf E and Ifergane, Shira and Freifeld, Oren},
  booktitle = {European Conference on Computer Vision},
  year      = {2024},
}
```
