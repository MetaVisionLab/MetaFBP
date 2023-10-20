
# MetaFBP: Learning to Learn High-Order Predictor for Personalized Facial Beauty Prediction
- 🔔This is the official (Pytorch) implementation for the paper "MetaFBP: Learning to Learn High-Order Predictor for Personalized Facial Beauty Prediction", ACM MM 2023.
- 🛖This repository is built on the [dragen1860/MAML-Pytorch](https://github.com/dragen1860/MAML-Pytorch). You can also view the project for details: [https://github.com/dragen1860/MAML-Pytorch](https://github.com/dragen1860/MAML-Pytorch)

# 🛠️Setup
## Runtime

The main python libraries we use:
- Python 3.8
- torch 1.8.1
- numpy 1.19.2

## Datasets
Please create a directory named `datasets` in current directory, then download these following datasets and unzip into `datasets`:

Download link: [GoogleDrive](https://drive.google.com/file/d/1kVqJyvrrJ1XqWWhY0yAWHM-ZzlHnnxFJ/view?usp=sharing) or [Quark](https://pan.quark.cn/s/aee85ace01e6)

You can also change the root directory of datasets by modifying the default value of the argument `--data-root` in [train_fea.py](train_fea.py)[L34], [train.py](train.py)[L40], and [test.py](test.py)[L59]:
```python
    # train_fea.py, Line 34
    parser.add_argument('--data-root', type=str, default='./datasets')
    # train.py, Line 40
    parser.add_argument('--data-root', type=str, default='./datasets')
    # test.py, Line 59
    parser.add_argument('--data-root', type=str, default='./datasets')
```


# 🎢Run
After finishing above steps, your directory structure of code may like this:
```text
MetaFBP/
    |–– data/
    |–– dataset/
        |–– FBP5500/
        |–– FBPSCUT/
        |–– US10K/
    |–– model/
    |–– util/
    README.md
    test.py
    test_fea.py
    train.py
    train.sh
    train_fea.py
    train_fea.sh
```
1. First of all, please train the universal feature extractor for each dataset:
    ```shell
    bash train_fea.sh PFBP-SCUT5500
    bash train_fea.sh PFBP-SCUT500
    bash train_fea.sh PFBP-US10K
    ```
    Usage of `train_fea.sh`:
    ```text
    bash train_fea.sh {arg1=dataset}
    ```
    - `dataset` specifies which dataset to train on, available ones are: `PFBP-SCUT5500`,`PFBP-SCUT500`, `PFBP-US10K`

2. Once the universal feature extractor is ready, you can run the experiments of PFBP task. For example, the following cmd runs the experiment of `MetaFBP-R` on PFBP-SCUT5500 benchmark with 5-way K-shot regression:
    ```shell
    bash train.sh MetaFBP-R PFBP-SCUT5500
    ```
    Usage of `train.sh`:
    ```text
    bash train.sh {arg1=model} {arg2=dataset}
    ```
    - `model` specifies which model to use, available ones are: `Base-MAML`,`MetaFBP-R`, `MetaFBP-T`
    - `dataset` specifies which dataset to train and test on, available ones are: `PFBP-SCUT5500`,`PFBP-SCUT500`, `PFBP-US10K`
   
# 📌Citation
If you would like to cite our work, the following bibtex code may be helpful:
```text
@inproceedings{lin2023metafbp,
    title={MetaFBP: Learning to Learn High-Order Predictor for Personalized Facial Beauty Prediction},
    author={Luojun Lin, Zhifeng Shen, Jia-Li Yin, Qipeng Liu, Yuanlong Yu, and Weijie Chen},
    booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
    year={2023},
}
```

# 🔗Acknowledgements
- Our code is built on dragen1860/MAML-Pytorch - https://github.com/dragen1860/MAML-Pytorch

# ⚖️License
This source code is released under the MIT license. View it [here](LICENSE)
