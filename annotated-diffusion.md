# The Annotated Diffusion Model

注释版扩散模型

> **注释版扩散模型** —— Hugging Face 官方博客对 DDPM（Denoising Diffusion Probabilistic Models）的逐步代码解读。
>
> - 原文链接：<https://huggingface.co/blog/annotated-diffusion>
> - 原文作者：Niels Rogge（nielsr）、Kashif Rasul（kashif）
> - 配套 Notebook：<https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/annotated_diffusion.ipynb>
> - 原文写作时间：2022 年 6 月 7 日；本文档整理自 Hugging Face 博客仓库原始 Markdown，图片链接已替换为绝对地址。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/annotated_diffusion.ipynb)

In this blog post, we'll take a deeper look into **Denoising Diffusion Probabilistic Models** (also known as DDPMs, diffusion models, score-based generative models or simply [autoencoders](https://benanne.github.io/2022/01/31/diffusion.html)) as researchers have been able to achieve remarkable results with them for (un)conditional image/audio/video generation. Popular examples (at the time of writing) include [GLIDE](https://arxiv.org/abs/2112.10741) and [DALL-E 2](https://openai.com/dall-e-2/) by OpenAI, [Latent Diffusion](https://github.com/CompVis/latent-diffusion) by the University of Heidelberg and [ImageGen](https://imagen.research.google/) by Google Brain.

在这篇博客中，我们将深入探讨**去噪扩散概率模型**，也称 DDPM、扩散模型、基于分数的生成模型，或干脆称为[自编码器](https://benanne.github.io/2022/01/31/diffusion.html)。研究人员已经利用这类模型，在有条件和无条件的图像、音频、视频生成任务上取得了显著成果。截至本文写作时，流行的例子包括 OpenAI 的 [GLIDE](https://arxiv.org/abs/2112.10741) 和 [DALL-E 2](https://openai.com/dall-e-2/)、海德堡大学的[潜空间扩散模型（Latent Diffusion）](https://github.com/CompVis/latent-diffusion)，以及 Google Brain 的 [ImageGen](https://imagen.research.google/)。


We'll go over the original DDPM paper by ([Ho et al., 2020](https://arxiv.org/abs/2006.11239)), implementing it step-by-step in PyTorch, based on Phil Wang's [implementation](https://github.com/lucidrains/denoising-diffusion-pytorch) - which itself is based on the [original TensorFlow implementation](https://github.com/hojonathanho/diffusion). Note that the idea of diffusion for generative modeling was actually already introduced in ([Sohl-Dickstein et al., 2015](https://arxiv.org/abs/1503.03585)). However, it took until ([Song et al., 2019](https://arxiv.org/abs/1907.05600)) (at Stanford University), and then ([Ho et al., 2020](https://arxiv.org/abs/2006.11239)) (at Google Brain) who independently improved the approach.

我们将讲解 DDPM 的原始论文（[Ho 等，2020](https://arxiv.org/abs/2006.11239)），并以 Phil Wang 的[实现](https://github.com/lucidrains/denoising-diffusion-pytorch)为基础，在 PyTorch 中逐步实现它；Phil Wang 的实现本身又基于[原始 TensorFlow 实现](https://github.com/hojonathanho/diffusion)。需要注意的是，将扩散用于生成建模的思想，实际上早在 [Sohl-Dickstein 等（2015）](https://arxiv.org/abs/1503.03585)的工作中就已提出。不过，直到斯坦福大学的 [Song 等（2019）](https://arxiv.org/abs/1907.05600)，以及随后 Google Brain 的 [Ho 等（2020）](https://arxiv.org/abs/2006.11239)，才分别独立地改进了这一方法。


Note that there are [several perspectives](https://twitter.com/sedielem/status/1530894256168222722?s=20&t=mfv4afx1GcNQU5fZklpACw) on diffusion models. Here, we employ the discrete-time (latent variable model) perspective, but be sure to check out the other perspectives as well.

理解扩散模型有[多种视角](https://twitter.com/sedielem/status/1530894256168222722?s=20&t=mfv4afx1GcNQU5fZklpACw)。本文采用离散时间的视角，即潜变量模型视角，但也请务必了解其他视角。


Alright, let's dive in!

好了，让我们开始吧！


```python
from IPython.display import Image
Image(filename='assets/78_annotated-diffusion/ddpm_paper.png')
```

![width-500](https://huggingface.co/blog/assets/78_annotated-diffusion/ddpm_paper.png)

We'll install and import the required libraries first (assuming you have [PyTorch](https://pytorch.org/) installed).

我们首先安装并导入所需的库，假设你已经安装了 [PyTorch](https://pytorch.org/)。


```python
!pip install -q -U einops datasets matplotlib tqdm

import math
from inspect import isfunction
from functools import partial

%matplotlib inline
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from einops import rearrange, reduce
from einops.layers.torch import Rearrange

import torch
from torch import nn, einsum
import torch.nn.functional as F
```

## What is a diffusion model?

什么是扩散模型？


A (denoising) diffusion model isn't that complex if you compare it to other generative models such as Normalizing Flows, GANs or VAEs: they all convert noise from some simple distribution to a data sample. This is also the case here where **a neural network learns to gradually denoise data** starting from pure noise. 

与归一化流（Normalizing Flows）、生成对抗网络（GAN）或变分自编码器（VAE）等其他生成模型相比，（去噪）扩散模型并没有那么复杂：这些模型都是将来自某种简单分布的噪声转换为数据样本。这里也是如此，**神经网络学习从纯噪声出发，逐步对数据去噪**。


In a bit more detail for images, the set-up consists of 2 processes:

更具体地说，对于图像，这一框架包含两个过程：

* a fixed (or predefined) forward diffusion process $q$ of our choosing, that gradually adds Gaussian noise to an image, until you end up with pure noise

  我们自行选择一个固定的或预先定义的正向扩散过程 $q$，逐步向图像添加高斯噪声，直到最终得到纯噪声。

* a learned reverse denoising diffusion process $p_\theta$, where a neural network is trained to gradually denoise an image starting from pure noise, until you end up with an actual image.

  一个通过学习得到的反向去噪扩散过程 $p_\theta$：训练神经网络，使其从纯噪声开始逐步对图像去噪，直到最终得到一张实际图像。


![width-600](https://huggingface.co/blog/assets/78_annotated-diffusion/diffusion_figure.png)

Both the forward and reverse process indexed by $t$ happen for some number of finite time steps $T$ (the DDPM authors use $T=1000$). You start with $t=0$ where you sample a real image $\mathbf{x}_0$ from your data distribution (let's say an image of a cat from ImageNet), and the forward process samples some noise from a Gaussian distribution at each time step $t$, which is added to the image of the previous time step. Given a sufficiently large $T$ and a well behaved schedule for adding noise at each time step, you end up with what is called an [isotropic Gaussian distribution](https://math.stackexchange.com/questions/1991961/gaussian-distribution-is-isotropic) at $t=T$ via a gradual process.

正向过程和反向过程都以 $t$ 为索引，在有限的 $T$ 个时间步内进行，DDPM 作者使用 $T=1000$。从 $t=0$ 开始，你从数据分布中采样一张真实图像 $\mathbf{x}_0$，例如 ImageNet 中的一张猫的图片。正向过程在每个时间步 $t$ 从高斯分布中采样噪声，并将其加到前一个时间步的图像上。当 $T$ 足够大，且每个时间步添加噪声的调度安排适当时，经过这一渐进过程，在 $t=T$ 时最终得到的分布就会是所谓的[各向同性高斯分布](https://math.stackexchange.com/questions/1991961/gaussian-distribution-is-isotropic)。


## In more mathematical form

更形式化的数学表述


Let's write this down more formally, as ultimately we need a tractable loss function which our neural network needs to optimize. 

让我们用更形式化的方式写下这个过程，因为最终需要一个可处理的损失函数，供神经网络进行优化。


Let $q(\mathbf{x}_0)$ be the real data distribution, say of "real images". We can sample from this distribution to get an image, $\mathbf{x}_0 \sim q(\mathbf{x}_0)$. We define the forward diffusion process $q(\mathbf{x}_t | \mathbf{x}_{t-1})$ which adds Gaussian noise at each time step $t$, according to a known variance schedule $0 < \beta_1 < \beta_2 < ... < \beta_T < 1$ as

令 $q(\mathbf{x}_0)$ 表示真实数据的分布，例如“真实图像”的分布。我们可以从这个分布中采样，得到图像 $\mathbf{x}_0 \sim q(\mathbf{x}_0)$。按照已知的方差调度 $0 < \beta_1 < \beta_2 < ... < \beta_T < 1$，我们定义在每个时间步 $t$ 添加高斯噪声的正向扩散过程 $q(\mathbf{x}_t | \mathbf{x}_{t-1})$ 如下：

$$
q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t \mathbf{I}). 
$$

Recall that a normal distribution (also called Gaussian distribution) is defined by 2 parameters: a mean $\mu$ and a variance $\sigma^2 \geq 0$. Basically, each new (slightly noisier) image at time step $t$ is drawn from a **conditional Gaussian distribution** with $\mathbf{\mu}_t = \sqrt{1 - \beta_t} \mathbf{x}_{t-1}$ and $\sigma^2_t = \beta_t$, which we can do by sampling $\mathbf{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ and then setting $\mathbf{x}_t = \sqrt{1 - \beta_t} \mathbf{x}_{t-1} +  \sqrt{\beta_t} \mathbf{\epsilon}$. 

回顾一下，正态分布，也称高斯分布，由两个参数定义：均值 $\mu$ 和方差 $\sigma^2 \geq 0$。基本上，时间步 $t$ 的每张新图像，噪声都会略多一些，且采样自一个**条件高斯分布**，其均值为 $\mathbf{\mu}_t = \sqrt{1 - \beta_t} \mathbf{x}_{t-1}$，方差为 $\sigma^2_t = \beta_t$。具体可以先采样 $\mathbf{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$，然后令 $\mathbf{x}_t = \sqrt{1 - \beta_t} \mathbf{x}_{t-1} + \sqrt{\beta_t} \mathbf{\epsilon}$，完成这一采样。


Note that the $\beta_t$ aren't constant at each time step $t$ (hence the subscript) --- in fact one defines a so-called **"variance schedule"**, which can be linear, quadratic, cosine, etc. as we will see further (a bit like a learning rate schedule). 

注意，不同时间步 $t$ 的 $\beta_t$ 并不是同一个常数，因此才有下标。实际上，我们会定义所谓的**“方差调度”**，它可以是线性的、二次的、余弦的等，后面会进一步介绍。这有点类似于学习率调度。


So starting from $\mathbf{x}_0$, we end up with $\mathbf{x}_1,  ..., \mathbf{x}_t, ..., \mathbf{x}_T$, where $\mathbf{x}_T$ is pure Gaussian noise if we set the schedule appropriately.

因此，从 $\mathbf{x}_0$ 出发，我们会依次得到 $\mathbf{x}_1, ..., \mathbf{x}_t, ..., \mathbf{x}_T$；如果调度设置适当，$\mathbf{x}_T$ 就是纯高斯噪声。


Now, if we knew the conditional distribution $p(\mathbf{x}_{t-1} | \mathbf{x}_t)$, then we could run the process in reverse: by sampling some random Gaussian noise $\mathbf{x}_T$, and then gradually "denoise" it so that we end up with a sample from the real distribution $\mathbf{x}_0$.

现在，如果我们知道条件分布 $p(\mathbf{x}_{t-1} | \mathbf{x}_t)$，就可以反向运行这个过程：先随机采样高斯噪声 $\mathbf{x}_T$，然后逐步对其“去噪”，最终得到来自真实分布的样本 $\mathbf{x}_0$。


However, we don't know $p(\mathbf{x}_{t-1} | \mathbf{x}_t)$. It's intractable since it requires knowing the distribution of all possible images in order to calculate this conditional probability. Hence, we're going to leverage a neural network to **approximate (learn) this conditional probability distribution**, let's call it $p_\theta (\mathbf{x}_{t-1} | \mathbf{x}_t)$, with $\theta$ being the parameters of the neural network, updated by gradient descent. 

然而，我们并不知道 $p(\mathbf{x}_{t-1} | \mathbf{x}_t)$。这个分布难以直接求解，因为计算这一条件概率需要知道所有可能图像的分布。因此，我们将利用神经网络来**近似（学习）这个条件概率分布**，并将其记为 $p_\theta (\mathbf{x}_{t-1} | \mathbf{x}_t)$，其中 $\theta$ 是通过梯度下降更新的神经网络参数。


Ok, so we need a neural network to represent a (conditional) probability distribution of the backward process. If we assume this reverse process is Gaussian as well, then recall that any Gaussian distribution is defined by 2 parameters:

也就是说，我们需要一个神经网络来表示反向过程的条件概率分布。如果假设这一反向过程也是高斯过程，那么请回想，任何高斯分布都由两个参数定义：


* a mean parametrized by $\mu_\theta$;

  由 $\mu_\theta$ 参数化的均值；

* a variance parametrized by $\Sigma_\theta$;

  由 $\Sigma_\theta$ 参数化的方差；


so we can parametrize the process as

因此，我们可以将这一过程参数化为：

$$ p_\theta (\mathbf{x}_{t-1} | \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \mu_\theta(\mathbf{x}_{t},t), \Sigma_\theta (\mathbf{x}_{t},t))$$
where the mean and variance are also conditioned on the noise level $t$.

其中，均值和方差也都以噪声水平 $t$ 为条件。


Hence, our neural network needs to learn/represent the mean and variance. However, the DDPM authors decided to **keep the variance fixed, and let the neural network only learn (represent) the mean $\mu_\theta$ of this conditional probability distribution**. From the paper:

因此，神经网络需要学习或表示均值和方差。不过，DDPM 作者决定**固定方差，只让神经网络学习（表示）这一条件概率分布的均值 $\mu_\theta$**。论文中的原话如下：


> First, we set $\Sigma_\theta ( \mathbf{x}_t, t) = \sigma^2_t \mathbf{I}$ to untrained time dependent constants. Experimentally, both $\sigma^2_t = \beta_t$ and $\sigma^2_t  = \tilde{\beta}_t$ (see paper) had similar results.

> 首先，我们将 $\Sigma_\theta (\mathbf{x}_t,t) = \sigma^2_t \mathbf{I}$ 设置为不参与训练、但随时间变化的常数。实验中，$\sigma^2_t = \beta_t$ 和 $\sigma^2_t = \tilde{\beta}_t$，详见论文，取得了相近的结果。


This was then later improved in the [Improved diffusion models](https://openreview.net/pdf?id=-NEXDKk8gZ) paper, where a neural network also learns the variance of this backwards process, besides the mean.

后来的[改进扩散模型（Improved diffusion models）](https://openreview.net/pdf?id=-NEXDKk8gZ)论文进一步改进了这一点：除了均值之外，神经网络还会学习这一反向过程的方差。


So we continue, assuming that our neural network only needs to learn/represent the mean of this conditional probability distribution.

接下来，我们继续假设神经网络只需要学习或表示这一条件概率分布的均值。


## Defining an objective function (by reparametrizing the mean)

定义目标函数（通过对均值重新参数化）


To derive an objective function to learn the mean of the backward process, the authors observe that the combination of $q$ and $p_\theta$ can be seen as a variational auto-encoder (VAE) [(Kingma et al., 2013)](https://arxiv.org/abs/1312.6114). Hence, the **variational lower bound** (also called ELBO) can be used to minimize the negative log-likelihood with respect to ground truth data sample $\mathbf{x}_0$ (we refer to the VAE paper for details regarding ELBO). It turns out that the ELBO for this process is a sum of losses at each time step $t$, $L = L_0 + L_1 + ... + L_T$. By construction of the forward $q$ process and backward process, each term (except for $L_0$) of the loss is actually the **KL divergence between 2 Gaussian distributions** which can be written explicitly as an L2-loss with respect to the means!

为了推导出学习反向过程均值的目标函数，作者指出，$q$ 与 $p_\theta$ 的组合可以看作一个变分自编码器（VAE）（[Kingma 等，2013](https://arxiv.org/abs/1312.6114)）。因此，可以利用**变分下界**，也称 ELBO，来最小化真实数据样本 $\mathbf{x}_0$ 的负对数似然，关于 ELBO 的细节请参阅 VAE 论文。结果表明，这一过程的 ELBO 可以表示为各时间步 $t$ 上损失项的总和，即 $L = L_0 + L_1 + ... + L_T$。根据正向过程 $q$ 和反向过程的构造，除 $L_0$ 外，每一项损失实际上都是**两个高斯分布之间的 KL 散度**，因而可以显式写成关于均值的 L2 损失！


A direct consequence of the constructed forward process $q$, as shown by Sohl-Dickstein et al., is that we can sample $\mathbf{x}_t$ at any arbitrary noise level conditioned on $\mathbf{x}_0$ (since sums of Gaussians is also Gaussian). This is very convenient:  we don't need to apply $q$ repeatedly in order to sample $\mathbf{x}_t$. 

正如 Sohl-Dickstein 等人所证明的，所构造的正向过程 $q$ 有一个直接推论：给定 $\mathbf{x}_0$，我们可以在任意噪声水平下直接采样 $\mathbf{x}_t$，因为高斯随机变量的和仍然服从高斯分布。这非常方便：为了采样 $\mathbf{x}_t$，我们不必反复应用 $q$。

We have that 

有如下关系：

$$q(\mathbf{x}_t | \mathbf{x}_0) = \cal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1- \bar{\alpha}_t) \mathbf{I})$$

with $\alpha_t := 1 - \beta_t$ and $\bar{\alpha}_t := \Pi_{s=1}^{t} \alpha_s$. Let's refer to this equation as the "nice property". This means we can sample Gaussian noise and scale it appropriately and add it to $\mathbf{x}_0$ to get $\mathbf{x}_t$ directly. Note that the $\bar{\alpha}_t$ are functions of the known $\beta_t$ variance schedule and thus are also known and can be precomputed. This then allows us, during training, to **optimize random terms of the loss function $L$** (or in other words, to randomly sample $t$ during training and optimize $L_t$).

其中 $\alpha_t := 1 - \beta_t$，$\bar{\alpha}_t := \Pi_{s=1}^{t} \alpha_s$。我们将这个等式称为“优良性质”（nice property）。它意味着，我们可以采样高斯噪声，对其适当缩放后加到 $\mathbf{x}_0$ 上，从而直接得到 $\mathbf{x}_t$。注意，$\bar{\alpha}_t$ 是已知方差调度 $\beta_t$ 的函数，因此也是已知的，可以预先计算。这使我们能够在训练时**优化损失函数 $L$ 中随机选取的项**，也就是说，在训练中随机采样 $t$，然后优化 $L_t$。


Another beauty of this property, as shown in Ho et al. is that one can (after some math, for which we refer the reader to [this excellent blog post](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)) instead **reparametrize the mean to make the neural network learn (predict) the added noise (via a network $\mathbf{\epsilon}_\theta(\mathbf{x}_t, t)$) for noise level $t$** in the KL terms which constitute the losses. This means that our neural network becomes a noise predictor, rather than a (direct) mean predictor. The mean can be computed as follows:

正如 Ho 等人所示，这个性质的另一个妙处在于，经过一些数学推导后，相关推导可参阅[这篇出色的博客](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)，可以在构成损失的 KL 散度项中，**对均值重新参数化，使神经网络学习（预测）噪声水平 $t$ 下所添加的噪声，具体通过网络 $\mathbf{\epsilon}_\theta(\mathbf{x}_t,t)$ 实现**。这意味着，我们的神经网络成为了噪声预测器，而不再是直接预测均值的预测器。均值可以按下式计算：


$$ \mathbf{\mu}_\theta(\mathbf{x}_t, t) = \frac{1}{\sqrt{\alpha_t}} \left(  \mathbf{x}_t - \frac{\beta_t}{\sqrt{1- \bar{\alpha}_t}} \mathbf{\epsilon}_\theta(\mathbf{x}_t, t) \right)$$

The final objective function $L_t$ then looks as follows (for a random time step $t$ given $\mathbf{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ ): 

于是，最终的目标函数 $L_t$ 如下所示，其中时间步 $t$ 随机选取，且 $\mathbf{\epsilon} \sim \mathcal{N}(\mathbf{0},\mathbf{I})$：


$$ \| \mathbf{\epsilon} - \mathbf{\epsilon}_\theta(\mathbf{x}_t, t) \|^2 = \| \mathbf{\epsilon} - \mathbf{\epsilon}_\theta( \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{(1- \bar{\alpha}_t)  } \mathbf{\epsilon}, t) \|^2.$$

Here, $\mathbf{x}_0$ is the initial (real, uncorrupted) image, and we see the direct noise level $t$ sample given by the fixed forward process. $\mathbf{\epsilon}$ is the pure noise sampled at time step $t$, and $\mathbf{\epsilon}_\theta (\mathbf{x}_t, t)$ is our neural network. The neural network is optimized using a simple mean squared error (MSE) between the true and the predicted Gaussian noise.

这里，$\mathbf{x}_0$ 是初始图像，即真实且未受噪声污染的图像；公式中也给出了由固定正向过程直接得到的噪声水平 $t$ 下的样本。$\mathbf{\epsilon}$ 是在时间步 $t$ 采样的纯噪声，$\mathbf{\epsilon}_\theta(\mathbf{x}_t,t)$ 则是我们的神经网络。神经网络通过最小化真实高斯噪声与预测高斯噪声之间简单的均方误差（MSE）进行优化。


The training algorithm now looks as follows:

现在，训练算法如下所示：


![width-400](https://huggingface.co/blog/assets/78_annotated-diffusion/training.png)

In other words:

换句话说：


* we take a random sample $\mathbf{x}_0$ from the real unknown and possibily complex data distribution $q(\mathbf{x}_0)$

  从真实、未知且可能复杂的数据分布 $q(\mathbf{x}_0)$ 中随机采样一个样本 $\mathbf{x}_0$。

* we sample a noise level $t$ uniformly between $1$ and $T$ (i.e., a random time step)

  在 $1$ 到 $T$ 之间均匀采样一个噪声水平 $t$，即随机选取一个时间步。

* we sample some noise from a Gaussian distribution and corrupt the input by this noise at level $t$ (using the nice property defined above)

  从高斯分布中采样噪声，并利用上面定义的优良性质，在噪声水平 $t$ 下用该噪声污染输入。

* the neural network is trained to predict this noise based on the corrupted image $\mathbf{x}_t$ (i.e. noise applied on $\mathbf{x}_0$ based on known schedule $\beta_t$)

  训练神经网络，使其根据受噪声污染的图像 $\mathbf{x}_t$ 预测这些噪声；这里的 $\mathbf{x}_t$ 是按照已知调度 $\beta_t$ 对 $\mathbf{x}_0$ 添加噪声后得到的。


In reality, all of this is done on batches of data, as one uses stochastic gradient descent to optimize neural networks.

实际中，这些操作都是以数据批次为单位进行的，因为我们使用随机梯度下降来优化神经网络。


## The neural network

神经网络


The neural network needs to take in a noised image at a particular time step and return the predicted noise. Note that the predicted noise is a tensor that has the same size/resolution as the input image. So technically, the network takes in and outputs tensors of the same shape. What type of neural network can we use for this?

神经网络需要接收某个特定时间步的含噪图像，并返回预测噪声。注意，预测噪声是一个与输入图像具有相同大小和分辨率的张量。因此，从技术上讲，网络输入和输出的张量形状相同。那么，我们可以使用什么类型的神经网络呢？


What is typically used here is very similar to that of an [Autoencoder](https://en.wikipedia.org/wiki/Autoencoder), which you may remember from typical "intro to deep learning" tutorials. Autoencoders have a so-called "bottleneck" layer in between the encoder and decoder. The encoder first encodes an image into a smaller hidden representation called the "bottleneck", and the decoder then decodes that hidden representation back into an actual image. This forces the network to only keep the most important information in the bottleneck layer.

这里通常使用的网络与[自编码器](https://en.wikipedia.org/wiki/Autoencoder)非常相似，你可能在常见的深度学习入门教程中见过它。自编码器在编码器和解码器之间有一个所谓的“瓶颈”层。编码器先将图像编码为更小的隐藏表示，也就是“瓶颈”；然后，解码器再将这一隐藏表示解码回实际图像。这迫使网络只在瓶颈层中保留最重要的信息。


In terms of architecture, the DDPM authors went for a **U-Net**, introduced by ([Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597)) (which, at the time, achieved state-of-the-art results for medical image segmentation). This network, like any autoencoder, consists of a bottleneck in the middle that makes sure the network learns only the most important information. Importantly, it introduced residual connections between the encoder and decoder, greatly improving gradient flow (inspired by ResNet in [He et al., 2015](https://arxiv.org/abs/1512.03385)).

在架构方面，DDPM 作者采用了由 [Ronneberger 等（2015）](https://arxiv.org/abs/1505.04597)提出的 **U-Net**，它当时在医学图像分割任务上取得了最先进的结果。与其他自编码器一样，这个网络中间也有一个瓶颈，用来确保网络只学习最重要的信息。重要的是，它在编码器和解码器之间引入了残差连接，大幅改善了梯度流动；这一设计受到 [He 等（2015）](https://arxiv.org/abs/1512.03385)的 ResNet 的启发。


![width-400](https://huggingface.co/blog/assets/78_annotated-diffusion/unet_architecture.jpg)

As can be seen, a U-Net model first downsamples the input (i.e. makes the input smaller in terms of spatial resolution), after which upsampling is performed.

可以看到，U-Net 模型先对输入进行下采样，即降低输入的空间分辨率，然后再进行上采样。


Below, we implement this network, step-by-step.

下面，我们将逐步实现这个网络。


### Network helpers

网络辅助函数与模块


First, we define some helper functions and classes which will be used when implementing the neural network. Importantly, we define a `Residual` module, which simply adds the input to the output of a particular function (in other words, adds a residual connection to a particular function).

首先，定义实现神经网络时会用到的一些辅助函数和类。其中重要的是 `Residual` 模块：它只是把输入加到某个函数的输出上，换句话说，就是为该函数添加残差连接。


We also define aliases for the up- and downsampling operations.

我们还会为上采样和下采样操作定义别名。


```python
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),
    )


def Downsample(dim, dim_out=None):
    # No More Strided Convolutions or Pooling
    return nn.Sequential(
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1),
    )
```

### Position embeddings

位置嵌入


As the parameters of the neural network are shared across time (noise level), the authors employ sinusoidal position embeddings to encode $t$, inspired by the Transformer ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)). This makes the neural network "know" at which particular time step (noise level) it is operating, for every image in a batch.

由于神经网络参数在不同时间步，也就是不同噪声水平之间共享，作者受 Transformer（[Vaswani 等，2017](https://arxiv.org/abs/1706.03762)）启发，使用正弦位置嵌入来编码 $t$。这样，对于批次中的每张图像，神经网络都能“知道”自己正在处理哪个特定时间步，也就是哪个噪声水平。


The `SinusoidalPositionEmbeddings` module takes a tensor of shape `(batch_size, 1)` as input (i.e. the noise levels of several noisy images in a batch), and turns this into a tensor of shape `(batch_size, dim)`, with `dim` being the dimensionality of the position embeddings. This is then added to each residual block, as we will see further.

`SinusoidalPositionEmbeddings` 模块接收形状为 `(batch_size, 1)` 的张量，表示一个批次中多张含噪图像各自的噪声水平，并将其转换为形状为 `(batch_size, dim)` 的张量，其中 `dim` 是位置嵌入的维度。随后，这些嵌入会被加入各个残差块中，后面会进一步看到这一点。


```python
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
```

### ResNet block

ResNet 残差块


Next, we define the core building block of the U-Net model. The DDPM authors employed a Wide ResNet block ([Zagoruyko et al., 2016](https://arxiv.org/abs/1605.07146)), but Phil Wang has replaced the standard convolutional layer by a "weight standardized" version, which works better in combination with group normalization (see ([Kolesnikov et al., 2019](https://arxiv.org/abs/1912.11370)) for details).

接下来，定义 U-Net 模型的核心构建块。DDPM 作者使用 Wide ResNet 残差块（[Zagoruyko 等，2016](https://arxiv.org/abs/1605.07146)），但 Phil Wang 将标准卷积层替换成了“权重标准化”版本。该版本与组归一化结合时效果更好，详情参阅 [Kolesnikov 等（2019）](https://arxiv.org/abs/1912.11370)。



```python
class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1 1", partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    """https://arxiv.org/abs/1512.03385"""

    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)
```

### Attention module

注意力模块


Next, we define the attention module, which the DDPM authors added in between the convolutional blocks. Attention is the building block of the famous Transformer architecture ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)), which has shown great success in various domains of AI, from NLP and vision to [protein folding](https://www.deepmind.com/blog/alphafold-a-solution-to-a-50-year-old-grand-challenge-in-biology). Phil Wang employs 2 variants of attention: one is regular multi-head self-attention (as used in the Transformer), the other one is a [linear attention variant](https://github.com/lucidrains/linear-attention-transformer) ([Shen et al., 2018](https://arxiv.org/abs/1812.01243)), whose time- and memory requirements scale linear in the sequence length, as opposed to quadratic for regular attention.

接下来，定义 DDPM 作者放在卷积块之间的注意力模块。注意力是著名的 Transformer 架构（[Vaswani 等，2017](https://arxiv.org/abs/1706.03762)）的基本构件，这一架构在自然语言处理、视觉乃至[蛋白质折叠](https://www.deepmind.com/blog/alphafold-a-solution-to-a-50-year-old-grand-challenge-in-biology)等人工智能领域都取得了巨大成功。Phil Wang 使用两种注意力变体：一种是常规的多头自注意力，即 Transformer 中使用的形式；另一种是[线性注意力变体](https://github.com/lucidrains/linear-attention-transformer)（[Shen 等，2018](https://arxiv.org/abs/1812.01243)）。后者的时间和内存需求随序列长度线性增长，而常规注意力则是二次增长。


For an extensive explanation of the attention mechanism, we refer the reader to Jay Allamar's [wonderful blog post](https://jalammar.github.io/illustrated-transformer/).

关于注意力机制的详细解释，请参阅 Jay Allamar 的[这篇精彩博客](https://jalammar.github.io/illustrated-transformer/)。


```python
class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), 
                                    nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)
```

### Group normalization

组归一化


The DDPM authors interleave the convolutional/attention layers of the U-Net with group normalization ([Wu et al., 2018](https://arxiv.org/abs/1803.08494)). Below, we define a `PreNorm` class, which will be used to apply groupnorm before the attention layer, as we'll see further. Note that there's been a [debate](https://tnq177.github.io/data/transformers_without_tears.pdf) about whether to apply normalization before or after attention in Transformers.

DDPM 作者在 U-Net 的卷积层和注意力层之间穿插使用组归一化（[Wu 等，2018](https://arxiv.org/abs/1803.08494)）。下面定义 `PreNorm` 类，用于在注意力层之前应用组归一化，后面会看到它的用法。注意，关于 Transformer 中应该在注意力之前还是之后应用归一化，一直存在[讨论](https://tnq177.github.io/data/transformers_without_tears.pdf)。


```python
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)
```

### Conditional U-Net

条件 U-Net


Now that we've defined all building blocks (position embeddings, ResNet blocks, attention and group normalization), it's time to define the entire neural network. Recall that the job of the network $\mathbf{\epsilon}_\theta(\mathbf{x}_t, t)$ is to take in a batch of noisy images and their respective noise levels, and output the noise added to the input. More formally:

现在，我们已经定义了所有构建块：位置嵌入、ResNet 残差块、注意力和组归一化，可以开始定义完整的神经网络了。回顾一下，网络 $\mathbf{\epsilon}_\theta(\mathbf{x}_t,t)$ 的任务是接收一批含噪图像及其各自的噪声水平，输出加到输入上的噪声。更形式化地说：


- the network takes a batch of noisy images of shape `(batch_size, num_channels, height, width)` and a batch of noise levels of shape `(batch_size, 1)` as input, and returns a tensor of shape `(batch_size, num_channels, height, width)`

  网络接收形状为 `(batch_size, num_channels, height, width)` 的一批含噪图像，以及形状为 `(batch_size, 1)` 的一批噪声水平，并返回形状为 `(batch_size, num_channels, height, width)` 的张量。


The network is built up as follows:

网络按以下方式构建：


* first, a convolutional layer is applied on the batch of noisy images, and position embeddings are computed for the noise levels

  首先，对这批含噪图像应用一个卷积层，并为噪声水平计算位置嵌入。

* next, a sequence of downsampling stages are applied. Each downsampling stage consists of 2 ResNet blocks + groupnorm + attention + residual connection + a downsample operation

  接着，依次执行多个下采样阶段。每个阶段包含两个 ResNet 残差块、组归一化、注意力、残差连接，以及一次下采样操作。

* at the middle of the network, again ResNet blocks are applied, interleaved with attention

  在网络中间，再次应用 ResNet 残差块，并在其中穿插注意力。

* next, a sequence of upsampling stages are applied. Each upsampling stage consists of 2 ResNet  blocks + groupnorm + attention + residual connection + an upsample operation

  然后，依次执行多个上采样阶段。每个阶段包含两个 ResNet 残差块、组归一化、注意力、残差连接，以及一次上采样操作。

* finally, a ResNet block followed by a convolutional layer is applied.

  最后，应用一个 ResNet 残差块，再接一个卷积层。


Ultimately, neural networks stack up layers as if they were lego blocks (but it's important to [understand how they work](http://karpathy.github.io/2019/04/25/recipe/)).

归根结底，神经网络就像搭乐高积木一样，将各层堆叠起来，但[理解它们的工作原理](http://karpathy.github.io/2019/04/25/recipe/)非常重要。



```python
class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        self_condition=False,
        resnet_block_groups=4,
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 1, padding=0) # changed to 1 and 0 from 7,3

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        time_dim = dim * 4

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Downsample(dim_in, dim_out)
                        if not is_last
                        else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Upsample(dim_out, dim_in)
                        if not is_last
                        else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        self.out_dim = default(out_dim, channels)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time, x_self_cond=None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)
```

## Defining the forward diffusion process

定义正向扩散过程


The forward diffusion process gradually adds noise to an image from the real distribution, in a number of time steps $T$. This happens according to a **variance schedule**. The original DDPM authors employed a linear schedule:

正向扩散过程会在 $T$ 个时间步内，逐渐向来自真实分布的图像添加噪声。这一过程按照某种**方差调度**进行。原始 DDPM 作者采用了线性调度：


> We set the forward process variances to constants
increasing linearly from $\beta_1 = 10^{−4}$
to $\beta_T = 0.02$.

> 我们将正向过程的方差设置为常数，这些常数从 $\beta_1 = 10^{−4}$ 到 $\beta_T = 0.02$ 线性递增。


However, it was shown in ([Nichol et al., 2021](https://arxiv.org/abs/2102.09672)) that better results can be achieved when employing a cosine schedule.

不过，[Nichol 等（2021）](https://arxiv.org/abs/2102.09672)表明，采用余弦调度可以获得更好的结果。


Below, we define various schedules for the $T$ timesteps (we'll choose one later on).

下面，我们为 $T$ 个时间步定义多种调度，稍后将从中选择一种。


```python
def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
```

To start with, let's use the linear schedule for $T=300$ time steps and define the various variables from the $\beta_t$ which we will need, such as the cumulative product of the variances $\bar{\alpha}_t$. Each of the variables below are just 1-dimensional tensors, storing values from $t$ to $T$. Importantly, we also define an `extract` function, which will allow us to extract the appropriate $t$ index for a batch of indices.

首先，我们对 $T=300$ 个时间步采用线性调度，并根据 $\beta_t$ 定义后续需要的各种变量，例如累积乘积 $\bar{\alpha}_t$。下面每个变量都是一维张量，存储从 $t$ 到 $T$ 各时间步的对应值。重要的是，我们还定义了 `extract` 函数，以便针对一批索引提取相应时间步 $t$ 的值。


```python
timesteps = 300

# define beta schedule
betas = linear_beta_schedule(timesteps=timesteps)

# define alphas 
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

# calculations for diffusion q(x_t | x_{t-1}) and others
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# calculations for posterior q(x_{t-1} | x_t, x_0)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
```

We'll illustrate with a cats image how noise is added at each time step of the diffusion process.

我们将用一张猫的图片，展示在扩散过程的各个时间步中如何添加噪声。


```python
from PIL import Image
import requests

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw) # PIL image of shape HWC
image
```
![width-400](https://huggingface.co/blog/assets/78_annotated-diffusion/output_cats.jpeg)

Noise is added to PyTorch tensors, rather than Pillow Images. We'll first define image transformations that allow us to go from a PIL image to a PyTorch tensor (on which we can add the noise), and vice versa.

噪声是加到 PyTorch 张量上的，而不是直接加到 Pillow 图像上。我们先定义图像变换，以便在 PIL 图像与 PyTorch 张量之间相互转换；转换为张量后，就可以添加噪声。


These transformations are fairly simple: we first normalize images by dividing by $255$ (such that they are in the $[0,1]$ range), and then make sure they are in the $[-1, 1]$ range. From the DPPM paper:

这些变换相当简单：首先将图像像素值除以 $255$，使其归一化到 $[0,1]$ 范围，然后再将其调整到 $[-1,1]$ 范围。DDPM 论文中写道：


> We assume that image data consists of integers in $\{0, 1, ... , 255\}$ scaled linearly to $[−1, 1]$. This
ensures that the neural network reverse process operates on consistently scaled inputs starting from
the standard normal prior $p(\mathbf{x}_T )$.

> 我们假设图像数据由 $\{0,1,...,255\}$ 范围内的整数组成，并将其线性缩放到 $[−1,1]$。这样可以确保神经网络的反向过程从标准正态先验 $p(\mathbf{x}_T)$ 出发，始终处理尺度一致的输入。



```python
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize

image_size = 128
transform = Compose([
    Resize(image_size),
    CenterCrop(image_size),
    ToTensor(), # turn into torch Tensor of shape CHW, divide by 255
    Lambda(lambda t: (t * 2) - 1),
    
])

x_start = transform(image).unsqueeze(0)
x_start.shape
```

**Output:**

**输出：**

```text
torch.Size([1, 3, 128, 128])
```

We also define the reverse transform, which takes in a PyTorch tensor containing values in $[-1, 1]$ and turn them back into a PIL image:

我们还定义了逆变换：它接收一个取值位于 $[-1,1]$ 范围内的 PyTorch 张量，并将其转换回 PIL 图像：


```python
import numpy as np

reverse_transform = Compose([
     Lambda(lambda t: (t + 1) / 2),
     Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
     Lambda(lambda t: t * 255.),
     Lambda(lambda t: t.numpy().astype(np.uint8)),
     ToPILImage(),
])
```

Let's verify this:

让我们验证一下：


```python
reverse_transform(x_start.squeeze())
```
    
![width-100](https://huggingface.co/blog/assets/78_annotated-diffusion/output_cats_verify.png)

We can now define the forward diffusion process as in the paper:

现在，我们可以按照论文定义正向扩散过程：



```python
# forward diffusion (using the nice property)
def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
```

Let's test it on a particular time step:

让我们在某个特定时间步上测试一下：


```python
def get_noisy_image(x_start, t):
  # add noise
  x_noisy = q_sample(x_start, t=t)

  # turn back into PIL image
  noisy_image = reverse_transform(x_noisy.squeeze())

  return noisy_image
```

```python
# take time step
t = torch.tensor([40])

get_noisy_image(x_start, t)
```

![width-100](https://huggingface.co/blog/assets/78_annotated-diffusion/output_cats_noisy.png)

Let's visualize this for various time steps:

让我们将不同时间步的结果可视化：


```python
import matplotlib.pyplot as plt

# use seed for reproducability
torch.manual_seed(0)

# source: https://pytorch.org/vision/stable/auto_examples/plot_transforms.html#sphx-glr-auto-examples-plot-transforms-py
def plot(imgs, with_orig=False, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0]) + with_orig
    fig, axs = plt.subplots(figsize=(200,200), nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        row = [image] + row if with_orig else row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if with_orig:
        axs[0, 0].set(title='Original image')
        axs[0, 0].title.set_size(8)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()
```

```python
plot([get_noisy_image(x_start, torch.tensor([t])) for t in [0, 50, 100, 150, 199]])
```
  
![width-800](https://huggingface.co/blog/assets/78_annotated-diffusion/output_cats_noisy_multiple.png)
    
This means that we can now define the loss function given the model as follows:

这意味着，给定模型后，我们现在可以按如下方式定义损失函数：


```python
def p_losses(denoise_model, x_start, t, noise=None, loss_type="l1"):
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)
    predicted_noise = denoise_model(x_noisy, t)

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss
```

The `denoise_model` will be our U-Net defined above. We'll employ the Huber loss between the true and the predicted noise.

`denoise_model` 就是前面定义的 U-Net。我们将使用真实噪声与预测噪声之间的 Huber 损失。


## Define a PyTorch Dataset + DataLoader

定义 PyTorch 数据集与数据加载器


Here we define a regular [PyTorch Dataset](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html). The dataset simply consists of images from a real dataset, like Fashion-MNIST, CIFAR-10 or ImageNet, scaled linearly to $[−1, 1]$.

这里定义一个常规的 [PyTorch 数据集](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)。数据集只需包含来自真实数据集的图像，例如 Fashion-MNIST、CIFAR-10 或 ImageNet，并将像素值线性缩放到 $[−1,1]$ 范围。


Each image is resized to the same size. Interesting to note is that images are also randomly horizontally flipped. From the paper:

每张图像都会被调整为相同大小。值得注意的是，图像还会被随机水平翻转。论文中写道：


> We used random horizontal flips during training for CIFAR10; we tried training both with and without flips, and found flips to improve sample quality slightly.

> 我们在 CIFAR10 的训练中使用了随机水平翻转；我们分别尝试了使用翻转和不使用翻转的训练方式，发现翻转能够略微改善样本质量。


Here we use the 🤗 [Datasets library](https://huggingface.co/docs/datasets/index) to easily load the Fashion MNIST dataset from the [hub](https://huggingface.co/datasets/fashion_mnist). This dataset consists of images which already have the same resolution, namely 28x28.

这里使用 Hugging Face 的 [Datasets 库](https://huggingface.co/docs/datasets/index)，方便地从 [Hub](https://huggingface.co/datasets/fashion_mnist) 加载 Fashion MNIST 数据集。这个数据集的图像本来就具有相同的分辨率，即 28×28。


```python
from datasets import load_dataset

# load dataset from the hub
dataset = load_dataset("fashion_mnist")
image_size = 28
channels = 1
batch_size = 128
```

Next, we define a function which we'll apply on-the-fly on the entire dataset. We use the `with_transform` [functionality](https://huggingface.co/docs/datasets/v2.2.1/en/package_reference/main_classes#datasets.Dataset.with_transform) for that. The function just applies some basic image preprocessing: random horizontal flips, rescaling and finally make them have values in the $[-1,1]$ range.

接下来定义一个函数，在使用整个数据集时动态应用它。为此，我们使用 `with_transform` [功能](https://huggingface.co/docs/datasets/v2.2.1/en/package_reference/main_classes#datasets.Dataset.with_transform)。这个函数只进行一些基本的图像预处理：随机水平翻转、重新缩放，最后将取值调整到 $[-1,1]$ 范围。


```python
from torchvision import transforms
from torch.utils.data import DataLoader

# define image transformations (e.g. using torchvision)
transform = Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
])

# define function
def transforms(examples):
   examples["pixel_values"] = [transform(image.convert("L")) for image in examples["image"]]
   del examples["image"]

   return examples

transformed_dataset = dataset.with_transform(transforms).remove_columns("label")

# create dataloader
dataloader = DataLoader(transformed_dataset["train"], batch_size=batch_size, shuffle=True)
```

```python
batch = next(iter(dataloader))
print(batch.keys())
```

**Output:**

**输出：**

输出：


```text
dict_keys(['pixel_values'])
```


## Sampling

采样


As we'll sample from the model during training (in order to track progress), we define the code for that below. Sampling is summarized in the paper as Algorithm 2:

由于我们会在训练期间从模型中采样，以便跟踪训练进展，因此先在下面定义相应代码。论文将采样过程概括为算法 2：


![width-500](https://huggingface.co/blog/assets/78_annotated-diffusion/sampling.png)

Generating new images from a diffusion model happens by reversing the diffusion process: we start from $T$, where we sample pure noise from a Gaussian distribution, and then use our neural network to gradually denoise it (using the conditional probability it has learned), until we end up at time step $t = 0$. As shown above, we can derive a slighly less denoised image $\mathbf{x}_{t-1 }$ by plugging in the reparametrization of the mean, using our noise predictor. Remember that the variance is known ahead of time.

利用扩散模型生成新图像，就是反向执行扩散过程：从时间步 $T$ 开始，先从高斯分布中采样纯噪声，然后利用神经网络，根据其学到的条件概率逐步去噪，直到时间步 $t=0$。如上所示，使用噪声预测器，将重新参数化后的均值代入，就能推导出噪声更少一些的图像 $\mathbf{x}_{t-1}$。请记住，方差是预先已知的。


Ideally, we end up with an image that looks like it came from the real data distribution.

理想情况下，最终得到的图像看起来就像是来自真实数据分布。


The code below implements this.

下面的代码实现了这一过程。


```python
@torch.no_grad()
def p_sample(model, x, t, t_index):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)
    
    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise 

# Algorithm 2 (including returning all images)
@torch.no_grad()
def p_sample_loop(model, shape):
    device = next(model.parameters()).device

    b = shape[0]
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)
    imgs = []

    for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
        imgs.append(img.cpu().numpy())
    return imgs

@torch.no_grad()
def sample(model, image_size, batch_size=16, channels=3):
    return p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))
```

Note that the code above is a simplified version of the original implementation. We found our simplification (which is in line with Algorithm 2 in the paper) to work just as well as the [original, more complex implementation](https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/diffusion_utils.py), which employs [clipping](https://github.com/hojonathanho/diffusion/issues/5).

注意，上面的代码是原始实现的简化版本。我们发现，这种与论文算法 2 一致的简化实现，与采用[裁剪操作](https://github.com/hojonathanho/diffusion/issues/5)的[原始、更复杂的实现](https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/diffusion_utils.py)效果同样好。


## Train the model

训练模型


Next, we train the model in regular PyTorch fashion. We also define some logic to periodically save generated images, using the `sample` method defined above.

接下来，按照常规的 PyTorch 方式训练模型。我们还会编写一些逻辑，使用上面定义的 `sample` 方法定期保存生成的图像。



```python
from pathlib import Path

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

results_folder = Path("./results")
results_folder.mkdir(exist_ok = True)
save_and_sample_every = 1000
```

Below, we define the model, and move it to the GPU. We also define a standard optimizer (Adam).

下面定义模型，并将其移动到 GPU 上。我们还会定义一个标准的优化器 Adam。


```python
from torch.optim import Adam

device = "cuda" if torch.cuda.is_available() else "cpu"

model = Unet(
    dim=image_size,
    channels=channels,
    dim_mults=(1, 2, 4,)
)
model.to(device)

optimizer = Adam(model.parameters(), lr=1e-3)
```

Let's start training!

开始训练吧！


```python
from torchvision.utils import save_image

epochs = 6

for epoch in range(epochs):
    for step, batch in enumerate(dataloader):
      optimizer.zero_grad()

      batch_size = batch["pixel_values"].shape[0]
      batch = batch["pixel_values"].to(device)

      # Algorithm 1 line 3: sample t uniformally for every example in the batch
      t = torch.randint(0, timesteps, (batch_size,), device=device).long()

      loss = p_losses(model, batch, t, loss_type="huber")

      if step % 100 == 0:
        print("Loss:", loss.item())

      loss.backward()
      optimizer.step()

      # save generated images
      if step != 0 and step % save_and_sample_every == 0:
        milestone = step // save_and_sample_every
        batches = num_to_groups(4, batch_size)
        all_images_list = list(map(lambda n: sample(model, batch_size=n, channels=channels), batches))
        all_images = torch.cat(all_images_list, dim=0)
        all_images = (all_images + 1) * 0.5
        save_image(all_images, str(results_folder / f'sample-{milestone}.png'), nrow = 6)
```

**Output:**

**输出：**

输出：


```text
Loss: 0.46477368474006653
Loss: 0.12143351882696152
Loss: 0.08106148988008499
Loss: 0.0801810547709465
Loss: 0.06122320517897606
Loss: 0.06310459971427917
Loss: 0.05681884288787842
Loss: 0.05729678273200989
Loss: 0.05497899278998375
Loss: 0.04439849033951759
Loss: 0.05415581166744232
Loss: 0.06020551547408104
Loss: 0.046830907464027405
Loss: 0.051029372960329056
Loss: 0.0478244312107563
Loss: 0.046767622232437134
Loss: 0.04305662214756012
Loss: 0.05216279625892639
Loss: 0.04748568311333656
Loss: 0.05107741802930832
Loss: 0.04588869959115982
Loss: 0.043014321476221085
Loss: 0.046371955424547195
Loss: 0.04952816292643547
Loss: 0.04472338408231735
```


## Sampling (inference)

采样（推理）


To sample from the model, we can just use our sample function defined above:

要从模型中采样，只需使用上面定义的采样函数：



```python
# sample 64 images
samples = sample(model, image_size=image_size, batch_size=64, channels=channels)

# show a random one
random_index = 5
plt.imshow(samples[-1][random_index].reshape(image_size, image_size, channels), cmap="gray")
```

![width-300](https://huggingface.co/blog/assets/78_annotated-diffusion/output.png)

Seems like the model is capable of generating a nice T-shirt! Keep in mind that the dataset we trained on is pretty low-resolution (28x28).

看来模型能够生成一件不错的 T 恤！请记住，我们用于训练的数据集分辨率相当低，只有 28×28。


We can also create a gif of the denoising process:

我们还可以把去噪过程制作成 GIF 动图：


```python
import matplotlib.animation as animation

random_index = 53

fig = plt.figure()
ims = []
for i in range(timesteps):
    im = plt.imshow(samples[i][random_index].reshape(image_size, image_size, channels), cmap="gray", animated=True)
    ims.append([im])

animate = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
animate.save('diffusion.gif')
plt.show()
```

![width-300](https://huggingface.co/blog/assets/78_annotated-diffusion/diffusion-sweater.gif)

## Follow-up reads

后续阅读


Note that the DDPM paper showed that diffusion models are a promising direction for (un)conditional image generation. This has since then (immensely) been improved, most notably for text-conditional image generation. Below, we list some important (but far from exhaustive) follow-up works:

DDPM 论文表明，扩散模型是有条件和无条件图像生成的一个很有前景的方向。此后，这一方向得到了极大的改进，尤其是在文本条件图像生成方面。下面列出了一些重要的后续工作，但远非完整清单：


- Improved Denoising Diffusion Probabilistic Models ([Nichol et al., 2021](https://arxiv.org/abs/2102.09672)): finds that learning the variance of the conditional distribution (besides the mean) helps in improving performance

  《改进的去噪扩散概率模型》（Nichol 等，2021）：发现除了均值之外，学习条件分布的方差也有助于提升性能。

- Cascaded Diffusion Models for High Fidelity Image Generation ([Ho et al., 2021](https://arxiv.org/abs/2106.15282)): introduces cascaded diffusion, which comprises a pipeline of multiple diffusion models that generate images of increasing resolution for high-fidelity image synthesis

  《用于高保真图像生成的级联扩散模型》（Ho 等，2021）：提出级联扩散，通过由多个扩散模型构成的流水线，生成分辨率逐渐提高的图像，实现高保真图像合成。

- Diffusion Models Beat GANs on Image Synthesis ([Dhariwal et al., 2021](https://arxiv.org/abs/2105.05233)): show that diffusion models can achieve image sample quality superior to the current state-of-the-art generative models by improving the U-Net architecture, as well as introducing classifier guidance

  《扩散模型在图像合成上超越 GAN》（Dhariwal 等，2021）：通过改进 U-Net 架构并引入分类器引导，证明扩散模型能够生成质量优于当时最先进生成模型的图像样本。

- Classifier-Free Diffusion Guidance ([Ho et al., 2021](https://openreview.net/pdf?id=qw8AKxfYbI)): shows that you don't need a classifier for guiding a diffusion model by jointly training a conditional and an unconditional diffusion model with a single neural network

  《无分类器扩散引导》（Ho 等，2021）：通过用同一个神经网络联合训练有条件和无条件扩散模型，证明无需分类器也可以对扩散模型进行引导。

- Hierarchical Text-Conditional Image Generation with CLIP Latents (DALL-E 2) ([Ramesh et al., 2022](https://cdn.openai.com/papers/dall-e-2.pdf)): uses a prior to turn a text caption into a CLIP image embedding, after which a diffusion model decodes it into an image

  《利用 CLIP 潜变量进行分层文本条件图像生成》（DALL-E 2；Ramesh 等，2022）：先使用先验模型将文本描述转换为 CLIP 图像嵌入，再由扩散模型将其解码为图像。

- Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding (ImageGen) ([Saharia et al., 2022](https://arxiv.org/abs/2205.11487)): shows that combining a large pre-trained language model (e.g. T5) with cascaded diffusion works well for text-to-image synthesis

  《具有深层语言理解能力的逼真文本到图像扩散模型》（ImageGen；Saharia 等，2022）：表明，将大型预训练语言模型（例如 T5）与级联扩散结合，能够很好地完成文本到图像合成任务。


Note that this list only includes important works until the time of writing, which is June 7th, 2022.

请注意，这份清单只包含截至本文写作时，也就是 2022 年 6 月 7 日的重要工作。


For now, it seems that the main (perhaps only) disadvantage of diffusion models is that they require multiple forward passes to generate an image (which is not the case for generative models like GANs). However, there's [research going on](https://arxiv.org/abs/2204.13902) that enables high-fidelity generation in as few as 10 denoising steps.

目前看来，扩散模型主要的、甚至可能是唯一的缺点，是生成一张图像需要进行多次前向传播，而 GAN 等生成模型并不需要。不过，已经有[相关研究](https://arxiv.org/abs/2204.13902)能够仅用 10 个去噪步骤实现高保真生成。



---

## 参考与延伸阅读

- 原博客：<https://huggingface.co/blog/annotated-diffusion>
- 配套 Colab Notebook：<https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/annotated_diffusion.ipynb>
- 本仓库的 [`denoising_diffusion_pytorch/`](./denoising_diffusion_pytorch) 目录即 Phil Wang 的 [`denoising-diffusion-pytorch`](https://github.com/lucidrains/denoising-diffusion-pytorch) 库（本文所解读实现的生产级版本）。
