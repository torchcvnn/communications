---
title: "torchcvnn: A PyTorch-based library to easily experiment with state-of-the-art Complex-Valued Neural Networks"
author: Jeremy Fix, Quentin Gabot, X. Huy Nguyen, Joana Frontera-Pons, Chengfang Ren and Jean-Philippe Ovarlez
institute: "CentraleSupelec"
format:
  revealjs:
    chalkboard: true
    fontsize: 22px
    controls: true
    logo: img/logo.png
    theme: [default]
    css: custom.css
    footer: Produced with [quarto](https://github.com/quarto-dev/quarto-cli)
bibliography: biblio.bib
highlight-style: pygments
---

# Introduction

## What is the point ?

Several domains involve complex valued data: **remote sensing** [@Barrachina2022], **MRI** [@Virtue2019],[@Solomon2024],[@Hemidi2023], **optics** [@Dinsdale2021], **computational neuroscience** [@Reichert2014]

Pytorch already implements complex valued gradient descent ([Wirtinger Calculus](https://pytorch.org/docs/stable/notes/autograd.html#autograd-for-complex-numbers)) but lacks several complex valued capabilities such as **datasets**, **cv-activations**, **layers**, **initialization functions**


. . . 

__Objective__: **robust, easy to use, complex valued neural networks + data loaders for PyTorch**. 

Existing frameworks :

- Tensorflow: [https://github.com/NEGU93/cvnn](https://github.com/NEGU93/cvnn) [@Barrachina2022]
- Pytorch: [complexPyTorch](https://github.com/wavefrontshaping/complexPyTorch) [@Matthes2021], [cplxmodule](https://github.com/ivannz/cplxmodule) [@Nazarov2019], [pytorch-complex](https://github.com/Roger-luo/pytorch-complex) (archived since 2019)

<!--

- to explore complex valued neural networks
- to provide a common implementation for benchmarking

-->


# Core components

## Datasets

:::: {.columns}
:::: {.column width="70%"}

Available [datasets](https://torchcvnn.github.io/torchcvnn/modules/datasets.html) :

- ALOS2/SLC formats
- Semantic segmentation : PolSF, Bretigny
- Classification : MSTAR/SAMPLE
- Reconstruction : CineMRI (MICCAI)

See [https://torchcvnn.github.io/torchcvnn/modules/datasets.html](https://torchcvnn.github.io/torchcvnn/modules/datasets.html)

```python
import torchcvnn
from torchcvnn.datasets.slc.dataset import SLCDataset

def get_pauli(data):
    # Returns Pauli in (H, W, C)
    HH = data["HH"]
    HV = data["HV"]
    VH = data["VH"]
    VV = data["VV"]

    alpha = HH + VV
    beta = HH - VV
    gamma = HV + VH

    return np.stack([beta, gamma, alpha], axis=-1)


patch_size = (3000, 3000)
dataset = SLCDataset(
    rootdir,
    transform=get_pauli,
    patch_size=patch_size,
)
```

::: 
:::: {.column width="30%"}

![](./img/datasets.png)

::: 
::: 

## Transforms

- Fourier `FFT`, Inverse Fourier `IFFT`

- Resize transforms : `SpatialResize`, `FFTResize` (crop/pad the FFT)

- `LogTransform` for converting the modulus to dB, keeping the phase

- Conversion Real $\leftrightarrow$ Complex : `ToReal`, `ToImaginary`,
  `RealImaginery` (split), `Amplitude`

- Transforms can be composed, as usual with pytorch

```python
import torchvision.transforms.v2 as v2
from torchcvnn.datasets import MSTARTargets
from torchcvnn.transforms import HWC2CHW, LogAmplitude, ToTensor, FFTResize

dataset = MSTARTargets(
	datadir,
	transform=v2.Compose(
		[
			HWC2CHW(),	
			FFTResize((opt.input_size, opt.input_size)),
			LogAmplitude(),	
			ToTensor('complex64'),
		]
	),
)
```


## Activation functions

Activation functions can be of different types :

- split activation functions, ndependently applied on both $\mathfrak{R}(z)$,
  $\mathfrak{I}(z)$ :
  CReLU, CPReLU, CTanh, ... `IndependentRealImag`

```python
from torchcvnn.nn import IndependentRealImag

CGELU = torchcvnn.IndependentRealImag(nn.GELU)
```


- taking into account both the magnitude and phase: Cardioid[@Virtue2019], Mod, modReLU,
  zAbsReLU, zLeakyReLU, zReLU

See [https://torchcvnn.github.io/torchcvnn/modules/nn.html#activations](https://torchcvnn.github.io/torchcvnn/modules/nn.html#activations)

## Initialization functions

Initialization is critical for successfull training of deep neural networks.

$$
\begin{eqnarray}
\label{eq:uglorot} \text{Glorot Uniform : } &  \mathcal{U}\left[-\displaystyle\frac{\sqrt{3}}{\sqrt{\text{fan}_{\text{in}} + \text{fan}_{\text{out}}}}, \frac{\sqrt{3}}{\sqrt{\text{fan}_{\text{in}} + \text{fan}_{\text{out}}}}\right]\\
\label{eq:nglorot} \text{Glorot Normal : } &  \mathcal{N}\left(0, \displaystyle\frac{1}{\sqrt{\text{fan}_{\text{in}}+ \text{fan}_{\text{out}}}}\right) \\
\label{eq:uhe} \text{He Uniform} &  w \sim \mathcal{U}\left[-\displaystyle\frac{\sqrt{3}}{\sqrt{\text{fan}_{\text{in}} }}, \frac{\sqrt{3}}{\sqrt{\text{fan}_{\text{in}} }}\right]\\
\label{eq:nhe} \text{He Normal} &  w \sim \mathcal{N}\left(0, \displaystyle\frac{1}{\sqrt{\text{fan}_{\text{in}} }}\right)
\end{eqnarray}
$$

## Pooling, dropout, normalization layers

- [Dropout layers](https://torchcvnn.github.io/torchcvnn/modules/nn.html#dropout-layers) : Dropout, Dropout2d
- [Pooling layers](https://torchcvnn.github.io/torchcvnn/modules/nn.html#pooling-layers) : MaxPool2d (on mod), AvgPool2d,
- [UpSampling layers](https://torchcvnn.github.io/torchcvnn/modules/nn.html#upsampling-layers) : ConvTranspose2d, Upsample
- [Normalization layers](https://torchcvnn.github.io/torchcvnn/modules/nn.html#normalization-layers) : BatchNorm{1d,2d}, LayerNorm, RMSNorm

	- **Complex valued Bath Normalization** [@Trabelsi2018]

	$$
	\begin{eqnarray}
	\nonumber \tilde{\mathbf{x}} &=& (\boldsymbol\Gamma\left(\mathbf{x})+\varepsilon\,\mathbf{I}\right)^{-\frac{1}{2}} \left(\mathbf{x} - \boldsymbol{\mu}(\mathbf{x})\right)\, ,\\
	\hat{\mathbf{x}} &=& \boldsymbol{\Lambda} \, \tilde{\mathbf{x}} + \boldsymbol{\beta}\, ,
	\end{eqnarray}
	$$

	- **LayerNorm** [@Ba2016] : statistics computed on the inputs of a layer, no need for
running statistics.

	- **RMSNorm** [@Zhang2019] : LayerNorm without centering

	$$
	\begin{eqnarray}
	\nonumber \tilde{\mathbf{x}} &=& (\boldsymbol\Gamma\left(\mathbf{x})+\varepsilon\,\mathbf{I}\right)^{-\frac{1}{2}} \mathbf{x} \, ,\\
	\hat{\mathbf{x}} &=& \boldsymbol{\Lambda} \, \tilde{\mathbf{x}}\, ,
	\end{eqnarray}
	$$



## Attention layers and transformers [1/2]

- Transformers [@Vaswani2017] introduced as an efficient FFNN for dealing with
  sequences. Extended to vision VIT [@dosovitskiy2021an]

- Fundamental component : Multi head attention module as scaled dot product : 

$$
\begin{equation*}
Att(\boldsymbol{Q}, \boldsymbol{K}, \boldsymbol{V}) = \mathrm{softmax}\left(\frac{\boldsymbol{Q} \, \boldsymbol{K}^T}{\sqrt{d}}\right) \, \boldsymbol{V}\, .
\end{equation*}
$$

- Extended to Complex by [@Eilers2023]

$$
\begin{equation*}
\mathbb{C}Att(\boldsymbol{Q}, \boldsymbol{K}, \boldsymbol{V}) = \mathrm{softmax}\left(\frac{\mathrm{Re} \left(\boldsymbol{Q} \, \boldsymbol{K}^H\right)}{\sqrt{d}}\right) \, \boldsymbol{V}\, .
\end{equation*}
$$

## Attention layers and transformers [2/2]

- Scaled Complex Valued ViTs in torchcvnn:<br/>


![](./img/scaled_vits.png)



```python
import torch.nn as nn
import torchcvnn.nn as c_nn
import torchcvnn.models as c_models

patch_embedder = nn.Sequential(
    c_nn.RMSNorm([cin, height, width]),
    nn.Conv2d(
        cin,
        hidden_dim,
        kernel_size=patch_size,
        stride=patch_size,
        dtype=torch.complex64,
    ),
    c_nn.RMSNorm([hidden_dim, height // patch_size, width // patch_size]),
)

vit_model = c_models.vit_b(patch_embedder)
# X is a torch tensor of dtype complex64
#                    and shape (B, C, H, W)
out = vit_model(X) 
# out is of dtype complex64
# and shape 
#   [B, hidden_dim, H//patch_size, W//patch_size]

```

# Use case : MSTAR classification with CV-CNNs and CV-ViTs

## Problem

:::: {.columns}
:::: {.column width="50%"}

- $\approx 14k$ images, $16$ classes, 
- $80\%$ for training, $20\%$ for validation, AdamW($\epsilon=0.003$, $\lambda=0.05$)
- image sizes $54\times 54 \rightarrow 193\times 193$  are FFT-resized
- magnitudes converted to dB scale, phase unchanged

:::
:::: {.column width="50%"}

![MSTAR distribution per class](./img/mstar_stats.png)

:::
:::

:::: {.columns}
:::: {.column width="50%"}


```python
import torchcvnn
from torchcvnn.datasets import MSTARTargets
from torchcvnn.transforms import HWC2CHW, LogAmplitude, ToTensor, FFTResize

size = 128

transform = v2.Compose(
    [HWC2CHW(), FFTResize((size, size)), 
	 LogAmplitude(), ToTensor('complex64')]
)

dataset = MSTARTargets(
    rootdir, transform=transform
)
X, y = dataset[0]


```


:::
:::: {.column width="50%"}

![MSTAR samples (magnitude)](./img/mstar_samples.png){width=45%}

:::
:::

Soure code : [https://github.com/torchcvnn/examples/tree/main/mstar_classification/](https://github.com/torchcvnn/examples/tree/main/mstar_classification/)

## Models and performances

:::: {.columns}
:::: {.column width="50%"}

- Pretrained real-valued resnet18 (e.g. timm) can be loaded and patched to complex

```python
complex_valued_model = convert_to_complex(resnet18())

def convert_to_complex(module: nn.Module) -> nn.Module:
    cdtype = torch.complex64
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            setattr(module, name, nn.Conv2d(..., dtype=cdtype))
        elif isinstance(child, nn.ReLU):
            setattr(module, name, c_nn.modReLU())
        elif isinstance(child, nn.BatchNorm2d):
            setattr(module, name, c_nn.BatchNorm2d(child.num_features))
        ....
        else:
            convert_to_complex(child)
```

:::
:::: {.column width="50%"}

![VIT architectures parameters](./img/vit_archis.png)

![Performances on the validation fold](./img/mstar_perf.png)

:::
:::

- Implement your own complex valued neural network, e.g. CV-VIT, see [mstar_classification/models.py:VisionTransformer](https://github.com/torchcvnn/examples/blob/9ba0042f9d7f0be911fe7fd26dc4b3479a85c0bd/mstar_classification/model.py#L291-L359)

	- embedding with an optional ConvSTEM (Hybrid VIT), on patches $16\times 16$
	- complex valued attention by [@Eilers2023]
	- additional class token, on which the classification head is connected



# Use case : PolSAR reconstruction with CV-AEs

## Problem

- Reconstruction of PolSAR (Quad pol) with complex valued auto-encoders with physical properties preservation [@Gabot2024]
- Full PolSF tile, non overlapping patches $64\times 64$, train($80\%$), valid ($20\%$)
- Encoder with $2\times$ Conv-BatchNorm-modReLU residual blocks, kernel size $3$, StridedConv
  downsampling
- Decoder with ConvTranspose upsampling, concat and $2$ residual blocks
- Trained with AdamW($\epsilon=0.0005$, $\lambda=0.0001$)

:::: {.columns}
:::: {.column width="50%"}

```python
from torchcvnn.datasets import ALOSDataset

crop_coordinates = ((2832, 736), (7888, 3520))
dataset = ALOSDataset(
    vol_filepath,
    patch_size=(512, 512),
    patch_stride=(128, 128),
    crop_coordinates=crop_coordinates,
)

```

:::
:::: {.column width="50%"}

![Complex valued Auto-encoder for PolSAR reconstruction](img/radar_ae.png){width=75%}

:::
:::

Source code : [https://github.com/QuentinGABOT/Reconstruction-PolSAR-Complex-AE](https://github.com/QuentinGABOT/Reconstruction-PolSAR-Complex-AE)

## Performances [1/2]

:::: {.columns}
:::: {.column width="50%"}

![](./img/radar_pauli.png){width=100%}

:::
:::: {.column width="50%"}

![](./img/radar_krogager.png){width=100%}

:::
:::

## Performances [2/2]

:::: {.columns}
:::: {.column width="50%"}

![](./img/radar_halpha.png){width=100%}

:::
:::: {.column width="50%"}

![](./img/radar_confusion.png){width=100%}

:::
:::

# Use case : Semantic segmentation with CV-UNet

## Problem

- ALOS2 [Polarimetric San Francisco](https://ietr-lab.univ-rennes1.fr/polsarpro-bio/san-francisco/),
- semantic segmentation with $6$ classes (+unlabel), $4000 \times 2000$
- non overlapping split, train($70\%$), valid($15\%$), test($15\%$)
- AdamW($\epsilon=0.001$, $\lambda=0.005$)

:::: {.columns}
:::: {.column width="50%"}

```python
import torchcvnn
from torchcvnn.datasets import PolSFDataset

def transform_patches(patches):
    # We keep all the patches and get the spectrum
    # from it
    # If you wish, you could filter out some polarizations
    # PolSF provides the four HH, HV, VH, VV
    patches = [np.abs(patchi) for _, patchi in patches.items()]
    return np.stack(patches)

dataset = PolSFDataset(rootdir, patch_size=((512, 512)), patch_stride=((512, 512)), transform=transform_patches
X, y = dataset[0]
```

:::
:::: {.column width="50%"}

![PolSF sample](./img/polsf_samples.png)

:::
:::

Source code : [https://github.com/torchcvnn/examples/tree/main/polsf_unet](https://github.com/torchcvnn/examples/tree/main/polsf_unet)

## Model and performances

- Complex valued UNet with $5$ encoder and $5$ decoder blocks, $52$M params
- Encoder with $2\times$ Conv-BatchNorm-modReLU residual blocks, kernel size $3$, StridedConv
  downsampling
- Decoder with bilinear upsampling, concat and $2$ residual blocks
- Shortcut connections between the encoder and decoder blocks

:::: {.columns}
:::: {.column width="40%"}

![Confusion matrix on the test fold](./img/polsf_confusion.png)

:::
:::: {.column width="20%"}

![Predicted segmentation mask](./img/polsf_pred.png)

:::
:::: {.column width="19%"}

![Ground truth labels](./img/polsf_gt.png)

:::
:::

# Use case : Neural Implicit Representation (NIR) for Cardiac reconstruction

## Problem

- In cardiac MRI, Fourier space is sampled by bands,
- more time is more bands is better resolution
- **Objective**:  reconstruct the image $\mathbf{I} \in \mathbb{C}^{N_x\times N_y}$ from the partially observed $k$-space $\mathbf{K} \in \mathbb{C}^{N_x\times N_y\times N_c}$ ($N_c$ coils)

- From CINEJense [@Hemidi2023] based on Instand Neural Graphic Primitives [@mueller2022]. No training, only inference.
- 2D+t input coordinates $(x, y, t)$ with real-valued multi-resolution hash encoding
- $2$ INR networks : image $\mathbf{I}_\theta^t(x,y)$ and coil's sensitivity $\mathbf{S}_\psi^{t,c}(x,y)$.
- INR = coordinates encoding + MLP (modReLU)

:::: {.columns}
:::: {.column width="45%"}

![Architecture of CINEJense [@Hemidi2023]](https://raw.githubusercontent.com/MDL-UzL/CineJENSE/refs/heads/main/images/CineJense_arch.png)

:::
:::: {.column width="45%"}

![Sample k-space, image and mask](./img/nir_samples.png)

:::
:::

Source code : [https://github.com/torchcvnn/examples/tree/main/nir_cinejense](https://github.com/torchcvnn/examples/tree/main/nir_cinejense)

## Model and performances

- Reconstruction loss with total variation regularizer
$$
\left(\hat{\theta}, \hat{\psi}\right)  =  \mbox{argmin}_{\theta, \psi} \frac{1}{N_cT}\sum_{\substack{c=0\\t=0}}^{\substack{T-1\\N_c-1}} L_\delta\left(\mathbf{M} \odot \mathcal{F}\left(\mathbf{I}_\theta^{t}\odot \mathbf{S}_\psi^{t,c}\right), \mathbf{K}^{t,c}\right) + \lambda \|\nabla \mathbf{I}_\theta^{t,c}\|_1\, ,
    \label{eq:cinejense}
$$

- The reconstruction combines the prediction with the partially observed k-space
  sampled arbitrarily over $X\times Y \times T$.

- Examples with acceleration factor $4$ (top), and $10$ (bottom)

![](https://raw.githubusercontent.com/torchcvnn/examples/main/nir_cinejense/gifs/acc4_sax_p002.gif){width=60%}

![](https://raw.githubusercontent.com/torchcvnn/examples/refs/heads/main/nir_cinejense/gifs/acc10_sax_p107.gif){width=60%}

# Conclusion

## Perspectives 

Two PhDs currently investigating :

- Complex valued generative models - Quentin Gabot, 
- Complex valued anomaly detection - Huy Nguyen, 


<!-- 

Two master students on :

- Modeling spiking neural networks with complex valued neural networks
- Self supervized pre-training

-->

More models, more datasets, more approaches

- Complex valued coordinates encoding (e.g. Hash encoding)
- Complex valued VAE : generative modeling
- Complex valued segmentation transformer : SegFormer
- Additional datasets supports : [S1SLC](https://ieee-dataport.org/open-access/s1slccvdl-complex-valued-annotated-single-look-complex-sentinel-1-sar-dataset-complex)

## Thanks

Thank you for your attention.

- Library available on [https://github.com/torchcvnn](https://github.com/torchcvnn),

>	```
>	pip install torchcvnn
>	```

- Examples available on [https://github.com/torchcvnn/examples](https://github.com/torchcvnn/examples),
- Documentation on [https://torchcvnn.github.io/torchcvnn/](https://torchcvnn.github.io/torchcvnn/),
- Code coverage on [https://torchcvnn.github.io/torchcvnn/htmlcov/](https://torchcvnn.github.io/torchcvnn/htmlcov/),
- Unit tests with pytest.

Join us in this effort, your contributions are welcome.

__Contact__: jeremy.fix@centralesupelec.fr

![](img/qr-code.png)

# References

## Bibliography
