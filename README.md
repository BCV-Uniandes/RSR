# Robust Super-Resolution

This is the official implementation of the paper: Generalized Real-World Super-Resolution through Adversarial Robustness.<br>

## Paper
[Generalized Real-World Super-Resolution through Adversarial Robustness](https://arxiv.org/pdf/2108.11505.pdf) <br/>
[Angela Castillo](https://angelacast135.github.io)<sup> 1*</sup>, [María Escobar](https://mc-escobar11.github.io)<sup> 1*</sup>, [Juan C. Pérez](https://juancprzs.github.io)<sup> 1, 2</sup>, [Andrés Romero](https://afromero.co/en)<sup> 3</sup>, [Radu Timofte](https://scholar.google.com/citations?user=u3MwH5kAAAAJ&hl=en)<sup> 3</sup>, [Luc Van Gool](https://scholar.google.com/citations?user=TwMib_QAAAAJ&hl=en)<sup> 3</sup>, [Pablo Arbeláez](https://scholar.google.com.co/citations?user=k0nZO90AAAAJ&hl=en)<sup>1</sup> <br/>
<sup>*</sup>Equal contribution.<br/>
<sup>1 </sup>Center for Research and Formation in Artificial Intelligence ([CinfonIA](https://cinfonia.uniandes.edu.co)), Universidad de Los Andes. <br/>
<sup>2 </sup>Image and Video Understanding Lab ([IVUL](https://cemse.kaust.edu.sa/ivul)), KAUST. <br/>
<sup>3 </sup>Computer Vision Lab ([CVL](https://www.vision.ee.ethz.ch/en/)), ETH Zürich. <br/>
<br/>

![](./figure1.png)

## Dependencies and Installation

- Python >= 3.7 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch == 1.6.0 TorchVision == 0.7.01.3](https://pytorch.org/)
- NVIDIA GPU + [CUDA v10.1.243](https://developer.nvidia.com/cuda-downloads)

1. Clone repo

    ```bash
    git clone https://github.com/BCV-Uniandes/RSR
    ```

1. Install dependent packages

    ```bash
    cd RSR
    pip install -r requirements.txt
    ```

1. Install the [BasicSR](https://github.com/xinntao/BasicSR) toolbox

    Please run the following commands in the **RSR root path** to install BasicSR:<br>
    (Make sure that your GCC version: gcc >= 5) <br>

    ```bash
    python setup.py develop --no_cuda_ext
    ```

<sub> BasicSR was only tested in Ubuntu. </sub>

## Dataset Preparation

- Please refer to [this web page](https://github.com/xinntao/BasicSR/blob/d21eac885b6de90a7adef7cc59e937dbdbb200b1/docs/DatasetPreparation.md#div2k) for details about the dataset organization and dataset augmentation.

## Train

- **Training command**: 

    ```bash
    bash train.sh
    ```
- **Pre-trained SR model**: Find the pre-trained SR model at [Drive](https://drive.google.com/file/d/1b3_bWZTjNO3iL2js1yWkJfjZykcQgvzT/view?usp=sharing).
- **Options/Configs**: Please check to [Config.md](https://github.com/xinntao/BasicSR/blob/d21eac885b6de90a7adef7cc59e937dbdbb200b1/docs/Config.md).
- **Logging**: Please refer to [Logging.md](https://github.com/xinntao/BasicSR/blob/d21eac885b6de90a7adef7cc59e937dbdbb200b1/docs/Logging.md).

## Pre-trained Model and Test

- Find [here](https://drive.google.com/drive/folders/1xtFRVrp2BHnOop9F_KO8i5aW1tVTLexD?usp=sharing) our pre-trained model. 
- **Test command**: 

    ```bash
    bash test.sh
    ```

## Citations

If RSR helps your research, please consider citing us.<br>

``` latex
@inproceedings{castillo2021generalized,
  title={Generalized Real-World Super-Resolution through Adversarial Robustness},
  author={Castillo, Angela and Escobar, Maria and P{\'e}rez, Juan C and Romero, Andr{\'e}s and Timofte, Radu and Van Gool, Luc and Arbelaez, Pablo},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={1855--1865},
  year={2021}
}
```

Find other resources in our [webpage](https://cinfonia.uniandes.edu.co/publications/generalized-real-world-super-resolution-through-adversarial-robustness/).

## License and Acknowledgement

This project borrows heavily from [BasicSR](https://github.com/xinntao/BasicSR/tree/d21eac885b6de90a7adef7cc59e937dbdbb200b1), we thank the authors for their contributions to the community.<br>
More details about **license** in [LICENSE](Lhttps://github.com/xinntao/BasicSR/blob/d21eac885b6de90a7adef7cc59e937dbdbb200b1/LICENSE/README.md).

## Contact

If you have any question, please email `a.castillo13@uniandes.edu.co`.
