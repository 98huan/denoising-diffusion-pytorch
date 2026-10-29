<img src="./images/denoising-diffusion.png" width="500px"></img>

## Denoising Diffusion Probabilistic Model, in Pytorch

> 《去噪扩散概率模型》（DDPM）的 PyTorch 实现。

Implementation of <a href="https://arxiv.org/abs/2006.11239">Denoising Diffusion Probabilistic Model</a> in Pytorch. It is a new approach to generative modeling that may <a href="https://ajolicoeur.wordpress.com/the-new-contender-to-gans-score-matching-with-langevin-sampling/">have the potential</a> to rival GANs. It uses denoising score matching to estimate the gradient of the data distribution, followed by Langevin sampling to sample from the true distribution.

> 这是 <a href="https://arxiv.org/abs/2006.11239">《去噪扩散概率模型》</a>（DDPM）的 PyTorch 实现。它是一种全新的生成建模方法，<a href="https://ajolicoeur.wordpress.com/the-new-contender-to-gans-score-matching-with-langevin-sampling/">有潜力</a>与 GAN 一较高下。其原理是：先用去噪得分匹配（denoising score matching）估计数据分布的梯度，再通过 Langevin 采样从真实分布中采样。

This implementation was inspired by the official Tensorflow version <a href="https://github.com/hojonathanho/diffusion">here</a>

> 本实现参考了官方 Tensorflow 版本<a href="https://github.com/hojonathanho/diffusion">代码</a>。

Youtube AI Educators - <a href="https://www.youtube.com/watch?v=W-O7AZNzbzQ">Yannic Kilcher</a> | <a href="https://www.youtube.com/watch?v=344w5h24-h8">AI Coffeebreak with Letitia</a> | <a href="https://www.youtube.com/watch?v=HoKDTa5jHvg">Outlier</a>

> YouTube AI 教育博主的讲解视频：<a href="https://www.youtube.com/watch?v=W-O7AZNzbzQ">Yannic Kilcher</a> | <a href="https://www.youtube.com/watch?v=344w5h24-h8">AI Coffeebreak with Letitia</a> | <a href="https://www.youtube.com/watch?v=HoKDTa5jHvg">Outlier</a>

<a href="https://github.com/yiyixuxu/denoising-diffusion-flax">Flax implementation</a> from <a href="https://github.com/yiyixuxu">YiYi Xu</a>

> 来自 <a href="https://github.com/yiyixuxu">YiYi Xu</a> 的 <a href="https://github.com/yiyixuxu/denoising-diffusion-flax">Flax 实现版本</a>。

<a href="https://huggingface.co/blog/annotated-diffusion">Annotated code</a> by Research Scientists / Engineers from <a href="https://huggingface.co/">🤗 Huggingface</a>

> <a href="https://huggingface.co/">🤗 Huggingface</a> 研究科学家/工程师编写的<a href="https://huggingface.co/blog/annotated-diffusion">带注释代码教程</a>（入门扩散模型的最佳材料）。

Update: Turns out none of the technicalities really matters at all | <a href="https://arxiv.org/abs/2208.09392">"Cold Diffusion" paper</a> | <a href="https://muse-model.github.io/">Muse</a>

> 更新：事实证明这些技术细节其实都不重要 | <a href="https://arxiv.org/abs/2208.09392">《Cold Diffusion》论文</a> | <a href="https://muse-model.github.io/">Muse</a>

<img src="./images/sample.png" width="500px"><img>

[![PyPI version](https://badge.fury.io/py/denoising-diffusion-pytorch.svg)](https://badge.fury.io/py/denoising-diffusion-pytorch)

## Install

> 安装

```bash
$ pip install denoising_diffusion_pytorch
```

## Usage

> 使用方法

> 最简用法：构建 `Unet` 噪声预测网络和 `GaussianDiffusion`，对图像张量计算扩散损失并反向传播；训练完成后调用 `sample` 生成图像。

```python
import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    flash_attn = True
)

diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    timesteps = 1000    # number of steps
)

training_images = torch.rand(8, 3, 128, 128) # images are normalized from 0 to 1
loss = diffusion(training_images)
loss.backward()

# after a lot of training

sampled_images = diffusion.sample(batch_size = 4)
sampled_images.shape # (4, 3, 128, 128)
```

Or, if you simply want to pass in a folder name and the desired image dimensions, you can use the `Trainer` class to easily train a model.

> 或者，如果你只想传入一个图片文件夹路径和目标图像尺寸，可以直接使用 `Trainer` 类轻松训练模型。

> 使用 `Trainer` 训练：只需传入图片文件夹路径，即可支持梯度累积、混合精度（AMP）、EMA、DDIM 快速采样与 FID 评估。

```python
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    flash_attn = True
)

diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    timesteps = 1000,           # number of steps
    sampling_timesteps = 250    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
)

trainer = Trainer(
    diffusion,
    'path/to/your/images',
    train_batch_size = 32,
    train_lr = 8e-5,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    calculate_fid = True              # whether to calculate fid during training
)

trainer.train()
```

Samples and model checkpoints will be logged to `./results` periodically

> 生成的样本图像和模型检查点（checkpoint）会定期保存到 `./results` 目录中。

## Explorative Modeling (Forward XM)

> 探索式建模（Forward XM）——训练时进行多候选损失计算

To use <a href="https://arxiv.org/abs/2607.27372">Explorative Modeling (Forward XM)</a> for multi-candidate loss calculation during training, wrap any diffusion model in `XMWrapper`.

> 若想在训练中使用 <a href="https://arxiv.org/abs/2607.27372">探索式建模（Forward XM）</a>进行多候选损失计算，只需用 `XMWrapper` 包装任意扩散模型。

> `XMWrapper` 用法：每个样本生成 4 个候选（对时间步/噪声多次采样）并取最小损失；采样接口 `sample` 与普通扩散模型完全一致。

```python
import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, XMWrapper

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8)
)

diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    timesteps = 1000
)

xm = XMWrapper(
    diffusion,
    candidates = 4 # generate 4 candidates per sample and pick minimum loss
)

training_images = torch.rand(8, 3, 128, 128)
loss = xm(training_images)
loss.backward()

# sampling works as usual

sampled_images = xm.sample(batch_size = 4)
```

## Multi-GPU Training

> 多 GPU 训练

The `Trainer` class is now equipped with <a href="https://huggingface.co/docs/accelerate/accelerator">🤗 Accelerator</a>. You can easily do multi-gpu training in two steps using their `accelerate` CLI

> `Trainer` 类现已集成 <a href="https://huggingface.co/docs/accelerate/accelerator">🤗 Accelerator</a>，借助其 `accelerate` 命令行工具，只需两步即可进行多 GPU 训练。

At the project root directory, where the training script is, run

> 第一步：在存放训练脚本的项目根目录下，运行以下命令进行加速环境配置（选择分布式策略、混合精度等）：

```python
$ accelerate config
```

Then, in the same directory

> 第二步：在同一目录下，运行以下命令启动分布式训练：

```python
$ accelerate launch train.py
```

## Miscellaneous

> 其他杂项

### 1D Sequence

> 1D 序列版本

By popular request, a 1D Unet + Gaussian Diffusion implementation.

> 应社区需求，这里提供了 1D Unet + 高斯扩散的实现。

> 1D 序列扩散用法：构造序列张量直接计算损失并训练；也可以用 `Dataset1D` 封装数据、交给 `Trainer1D` 训练，完成后采样。

```python
import torch
from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D

model = Unet1D(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    channels = 32
)

diffusion = GaussianDiffusion1D(
    model,
    seq_length = 128,
    timesteps = 1000,
    objective = 'pred_v'
)

training_seq = torch.rand(64, 32, 128) # features are normalized from 0 to 1

loss = diffusion(training_seq)
loss.backward()

# Or using trainer

dataset = Dataset1D(training_seq)  # this is just an example, but you can formulate your own Dataset and pass it into the `Trainer1D` below

trainer = Trainer1D(
    diffusion,
    dataset = dataset,
    train_batch_size = 32,
    train_lr = 8e-5,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
)
trainer.train()

# after a lot of training

sampled_seq = diffusion.sample(batch_size = 4)
sampled_seq.shape # (4, 32, 128)

```

`Trainer1D` does not evaluate the generated samples in any way since the type of data is not known.

> `Trainer1D` 不会对生成的样本做任何评估，因为 1D 数据的类型是未知的。

You could consider adding a suitable metric to the training loop yourself after doing an editable install of this package
`pip install -e .`.

> 你可以考虑在以可编辑模式安装本包（`pip install -e .`）之后，自行在训练循环中添加合适的评估指标。

## Citations

> 引用（本实现所依据的论文清单）

> 《去噪扩散概率模型》（DDPM 原始论文，本仓库代码与其一一对应）。

```bibtex
@inproceedings{NEURIPS2020_4c5bcfec,
    author      = {Ho, Jonathan and Jain, Ajay and Abbeel, Pieter},
    booktitle   = {Advances in Neural Information Processing Systems},
    editor      = {H. Larochelle and M. Ranzato and R. Hadsell and M.F. Balcan and H. Lin},
    pages       = {6840--6851},
    publisher   = {Curran Associates, Inc.},
    title       = {Denoising Diffusion Probabilistic Models},
    url         = {https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf},
    volume      = {33},
    year        = {2020}
}
```

> 《改进的去噪扩散概率模型》（引入学习方差、余弦噪声调度等改进，对应 cosine schedule）。

```bibtex
@InProceedings{pmlr-v139-nichol21a,
    title       = {Improved Denoising Diffusion Probabilistic Models},
    author      = {Nichol, Alexander Quinn and Dhariwal, Prafulla},
    booktitle   = {Proceedings of the 38th International Conference on Machine Learning},
    pages       = {8162--8171},
    year        = {2021},
    editor      = {Meila, Marina and Zhang, Tong},
    volume      = {139},
    series      = {Proceedings of Machine Learning Research},
    month       = {18--24 Jul},
    publisher   = {PMLR},
    pdf         = {http://proceedings.mlr.press/v139/nichol21a/nichol21a.pdf},
    url         = {https://proceedings.mlr.press/v139/nichol21a.html},
}
```

> 《用扩散模型进行密度估计》（v 参数化的出处，对应 `objective='pred_v'`）。

```bibtex
@inproceedings{kingma2021on,
    title       = {On Density Estimation with Diffusion Models},
    author      = {Diederik P Kingma and Tim Salimans and Ben Poole and Jonathan Ho},
    booktitle   = {Advances in Neural Information Processing Systems},
    editor      = {A. Beygelzimer and Y. Dauphin and P. Liang and J. Wortman Vaughan},
    year        = {2021},
    url         = {https://openreview.net/forum?id=2LdBqxc1Yv}
}
```

> 《阐明扩散生成模型的设计空间》（EDM，对应 `elucidated_diffusion.py`）。

```bibtex
@article{Karras2022ElucidatingTD,
    title   = {Elucidating the Design Space of Diffusion-Based Generative Models},
    author  = {Tero Karras and Miika Aittala and Timo Aila and Samuli Laine},
    journal = {ArXiv},
    year    = {2022},
    volume  = {abs/2206.00364}
}
```

> 《去噪扩散隐式模型》（DDIM，跳步快速采样，对应 `sampling_timesteps`）。

```bibtex
@article{Song2021DenoisingDI,
    title   = {Denoising Diffusion Implicit Models},
    author  = {Jiaming Song and Chenlin Meng and Stefano Ermon},
    journal = {ArXiv},
    year    = {2021},
    volume  = {abs/2010.02502}
}
```

> 《Analog Bits：用带自条件的扩散模型生成离散数据》（self-conditioning 的出处）。

```bibtex
@misc{chen2022analog,
    title   = {Analog Bits: Generating Discrete Data using Diffusion Models with Self-Conditioning},
    author  = {Ting Chen and Ruixiang Zhang and Geoffrey Hinton},
    year    = {2022},
    eprint  = {2208.04202},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```

> 《用于扩散模型快速采样的渐进蒸馏》。

```bibtex
@article{Salimans2022ProgressiveDF,
    title   = {Progressive Distillation for Fast Sampling of Diffusion Models},
    author  = {Tim Salimans and Jonathan Ho},
    journal = {ArXiv},
    year    = {2022},
    volume  = {abs/2202.00512}
}
```

> 《无分类器引导的扩散模型》（CFG，对应 `classifier_free_guidance.py`）。

```bibtex
@article{Ho2022ClassifierFreeDG,
    title   = {Classifier-Free Diffusion Guidance},
    author  = {Jonathan Ho},
    journal = {ArXiv},
    year    = {2022},
    volume  = {abs/2207.12598}
}
```

> 《不再使用步长卷积或池化：面向低分辨率图像和小目标的新 CNN 构建块》（对应代码中的 `Downsample` 实现）。

```bibtex
@article{Sunkara2022NoMS,
    title   = {No More Strided Convolutions or Pooling: A New CNN Building Block for Low-Resolution Images and Small Objects},
    author  = {Raja Sunkara and Tie Luo},
    journal = {ArXiv},
    year    = {2022},
    volume  = {abs/2208.03641}
}
```

> 《迭代生成的可扩展自适应计算》。

```bibtex
@inproceedings{Jabri2022ScalableAC,
    title   = {Scalable Adaptive Computation for Iterative Generation},
    author  = {A. Jabri and David J. Fleet and Ting Chen},
    year    = {2022}
}
```

> 《DPM-Solver++：扩散概率模型引导采样的快速求解器》。

```bibtex
@article{Cheng2022DPMSolverPlusPlus,
    title   = {DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models},
    author  = {Cheng Lu and Yuhao Zhou and Fan Bao and Jianfei Chen and Chongxuan Li and Jun Zhu},
    journal = {NeuRips 2022 Oral},
    year    = {2022},
    volume  = {abs/2211.01095}
}
```

> 《simple diffusion：面向高分辨率图像的端到端扩散》（对应 `simple_diffusion.py`）。

```bibtex
@inproceedings{Hoogeboom2023simpleDE,
    title   = {simple diffusion: End-to-end diffusion for high resolution images},
    author  = {Emiel Hoogeboom and Jonathan Heek and Tim Salimans},
    year    = {2023}
}
```

> 《Dual PatchNorm》（对应 `DualPatchNorm` 归一化实现）。

```bibtex
@misc{https://doi.org/10.48550/arxiv.2302.01327,
    doi     = {10.48550/ARXIV.2302.01327},
    url     = {https://arxiv.org/abs/2302.01327},
    author  = {Kumar, Manoj and Dehghani, Mostafa and Houlsby, Neil},
    title   = {Dual PatchNorm},
    publisher = {arXiv},
    year    = {2023},
    copyright = {Creative Commons Attribution 4.0 International}
}
```

> 《基于 Min-SNR 加权策略的高效扩散训练》（对应 `min_snr_loss_weight` 参数）。

```bibtex
@inproceedings{Hang2023EfficientDT,
    title   = {Efficient Diffusion Training via Min-SNR Weighting Strategy},
    author  = {Tiankai Hang and Shuyang Gu and Chen Li and Jianmin Bao and Dong Chen and Han Hu and Xin Geng and Baining Guo},
    year    = {2023}
}
```

> 《带偏移噪声的扩散》（offset noise 技术博客，对应 `offset_noise_strength` 参数）。

```bibtex
@misc{Guttenberg2023,
    author  = {Nicholas Guttenberg},
    url     = {https://www.crosslabs.org/blog/diffusion-with-offset-noise}
}
```

> 《常见的扩散噪声调度与采样步数存在缺陷》。

```bibtex
@inproceedings{Lin2023CommonDN,
    title   = {Common Diffusion Noise Schedules and Sample Steps are Flawed},
    author  = {Shanchuan Lin and Bingchen Liu and Jiashi Li and Xiao Yang},
    year    = {2023}
}
```

> 《FlashAttention：具备 IO 感知的快速、内存高效精确注意力》（对应 `flash_attn` 参数）。

```bibtex
@inproceedings{dao2022flashattention,
    title   = {Flash{A}ttention: Fast and Memory-Efficient Exact Attention with {IO}-Awareness},
    author  = {Dao, Tri and Fu, Daniel Y. and Ermon, Stefano and Rudra, Atri and R{\'e}, Christopher},
    booktitle = {Advances in Neural Information Processing Systems},
    year    = {2022}
}
```

> 《可量化 Transformer：通过让注意力头"无为"来消除异常值》（对应 `Attend` 中的可量化记忆 kv 机制）。

```bibtex
@article{Bondarenko2023QuantizableTR,
    title   = {Quantizable Transformers: Removing Outliers by Helping Attention Heads Do Nothing},
    author  = {Yelysei Bondarenko and Markus Nagel and Tijmen Blankevoort},
    journal = {ArXiv},
    year    = {2023},
    volume  = {abs/2306.12929},
    url     = {https://api.semanticscholar.org/CorpusID:259224568}
}
```

> 《分析与改进扩散模型的训练动态》。

```bibtex
@article{Karras2023AnalyzingAI,
    title   = {Analyzing and Improving the Training Dynamics of Diffusion Models},
    author  = {Tero Karras and Miika Aittala and Jaakko Lehtinen and Janne Hellsten and Timo Aila and Samuli Laine},
    journal = {ArXiv},
    year    = {2023},
    volume  = {abs/2312.02696},
    url     = {https://api.semanticscholar.org/CorpusID:265659032}
}
```

> 《互斥扩散：通过噪声分配加速扩散训练》（对应 `immiscible` 参数）。

```bibtex
@article{Li2024ImmiscibleDA,
    title   = {Immiscible Diffusion: Accelerating Diffusion Training with Noise Assignment},
    author  = {Yiheng Li and Heyang Jiang and Akio Kodaira and Masayoshi Tomizuka and Kurt Keutzer and Chenfeng Xu},
    journal = {ArXiv},
    year    = {2024},
    volume  = {abs/2406.12303},
    url     = {https://api.semanticscholar.org/CorpusID:270562607}
}
```

> 《CFG++：面向扩散模型的流形约束无分类器引导》。

```bibtex
@article{Chung2024CFGMC,
    title   = {CFG++: Manifold-constrained Classifier Free Guidance for Diffusion Models},
    author  = {Hyungjin Chung and Jeongsol Kim and Geon Yeong Park and Hyelin Nam and Jong Chul Ye},
    journal = {ArXiv},
    year    = {2024},
    volume  = {abs/2406.08070},
    url     = {https://api.semanticscholar.org/CorpusID:270391454}
}
```

> 《消除扩散模型中高引导尺度的过饱和与伪影》。

```bibtex
@inproceedings{Sadat2024EliminatingOA,
    title   = {Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models},
    author  = {Seyedmorteza Sadat and Otmar Hilliges and Romann M. Weber},
    year    = {2024},
    url     = {https://api.semanticscholar.org/CorpusID:273098845}
}
```

> 《探索式建模：解锁第三个预训练维度与端到端生成》（Forward XM 论文，对应 `XMWrapper`）。

```bibtex
@misc{gladstone2026explorativemodelingunlockingpretraining,
    title   = {Explorative Modeling: Unlocking a Third Pretraining Axis and End-to-End Generation},
    author  = {Alexi Gladstone and Heng Ji and Yilun Du},
    year    = {2026},
    eprint  = {2607.27372},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url     = {https://arxiv.org/abs/2607.27372},
}
```
