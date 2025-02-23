# Trainable Highly-expressive Activation Functions [ECCV 2024]

[Irit Chelly*](https://irita42.wixsite.com/mysite), [Shahaf E. Finder*](https://shahaffind.github.io/), [Shira Ifergane](https://www.linkedin.com/in/shira-ifergane/), and [Oren Freifeld](https://www.cs.bgu.ac.il/~orenfr/).

[![arXiv](https://img.shields.io/badge/arXiv-2407.07564-b31b1b.svg?style=flat)](https://arxiv.org/abs/2407.07564)

Pytorch implementation of DiTAC.

<br>
<p align="center">
<img src="https://github.com/BGU-CS-VIL/DiTAC/blob/main/.github/ditac_fig.png" alt="DiTAC typical results" width="520" height="320">
</p>

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
If you find this repository helpful, please consider citing our paper:
```
@inproceedings{Chelly:ECCV:2024DiTAC,
  title     = {Trainable Highly-expressive Activation Functions},
  author    = {Chelly, Irit and Finder, Shahaf E and Ifergane, Shira and Freifeld, Oren},
  booktitle = {European Conference on Computer Vision},
  year      = {2024},
}
```
Please also consider citing previous works:
```
@inproceedings{martinez2022closed,
  title={Closed-form diffeomorphic transformations for time series alignment},
  author={Martinez, I{\~n}igo and Viles, Elisabeth and Olaizola, Igor G},
  booktitle={International Conference on Machine Learning},
  pages={15122--15158},
  year={2022},
  organization={PMLR}
}

@article{freifeld2017transformations,
  title={Transformations based on continuous piecewise-affine velocity fields},
  author={Freifeld, Oren and Hauberg, S{\o}ren and Batmanghelich, Kayhan and Fisher, Jonn W},
  journal={IEEE transactions on pattern analysis and machine intelligence},
  volume={39},
  number={12},
  pages={2496--2509},
  year={2017},
  publisher={IEEE}
}
```
