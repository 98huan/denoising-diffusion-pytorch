# The Illustrated Transformer

## 图解 Transformer

> **原文**：<https://jalammar.github.io/illustrated-transformer/>
> **作者**：Jay Alammar（2018 年 6 月 27 日发布；含 2020 年 7 月与 2025 年更新）
> **许可**：CC BY-NC-SA 4.0
> **说明**：本文是全文双语整理版——每个英文段落之后紧跟对应的完整中文翻译，图片保留原站外链（`jalammar.github.io/images/t/`）。
> **原文页脚信息**：讨论见 Hacker News、Reddit r/MachineLearning；官方页面列有阿拉伯语、简体中文、法语、意大利语、日语、韩语、波斯语、俄语、西班牙语、越南语等译本；MIT 深度学习课程亦引用本文。

---

**Update:** This post has now become a book! Check out [LLM-book.com](https://llm-book.com) which contains (Chapter 3) an updated and expanded version of this post speaking about the latest Transformer models and how they've evolved in the seven years since the original Transformer (like Multi-Query Attention and RoPE Positional embeddings).

**更新：** 这篇文章现在已经成为一本书了！欢迎访问 LLM-book.com，其中第 3 章是本文的更新与扩充版，讲述了最新的 Transformer 模型，以及自原始 Transformer 诞生七年来它们是如何演进的（例如 Multi-Query Attention 和 RoPE 位置嵌入）。

In the [previous post, we looked at Attention](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/) – a ubiquitous method in modern deep learning models. Attention is a concept that helped improve the performance of neural machine translation applications. In this post, we will look at **The Transformer** – a model that uses attention to boost the speed with which these models can be trained. The Transformer outperforms the Google Neural Machine Translation model in specific tasks. The biggest benefit, however, comes from how The Transformer lends itself to parallelization. It is in fact Google Cloud’s recommendation to use The Transformer as a reference model to use their [Cloud TPU](https://cloud.google.com/tpu/) offering. So let’s try to break the model apart and look at how it functions.

在[上一篇文章中，我们讨论了注意力（Attention）](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)——一种在现代深度学习模型中无处不在的方法。注意力这一概念帮助提升了神经机器翻译应用的表现。本文中，我们将研究 **The Transformer**——一个利用注意力来提升训练速度的模型。在特定任务上，Transformer 的表现优于 Google 神经机器翻译模型。不过，它最大的优势在于天生适合并行化。事实上，Google Cloud 就推荐把 Transformer 作为使用其 [Cloud TPU](https://cloud.google.com/tpu/) 的参考模型。那么，我们就试着把这个模型拆开，看看它是如何运作的。

The Transformer was proposed in the paper [Attention is All You Need](https://arxiv.org/abs/1706.03762). A TensorFlow implementation of it is available as a part of the [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor) package. Harvard’s NLP group created a [guide annotating the paper with PyTorch implementation](http://nlp.seas.harvard.edu/2018/04/03/attention.html). In this post, we will attempt to oversimplify things a bit and introduce the concepts one by one to hopefully make it easier to understand to people without in-depth knowledge of the subject matter.

Transformer 由论文 [Attention is All You Need](https://arxiv.org/abs/1706.03762) 提出，其 TensorFlow 实现作为 [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor) 包的一部分提供。哈佛大学的 NLP 组编写了一份[逐段注解论文并附 PyTorch 实现的指南](http://nlp.seas.harvard.edu/2018/04/03/attention.html)。在本文中，我们会尽量把内容讲得简单一些，一个一个地引入概念，希望能让没有深厚背景知识的读者也更容易理解。

**2025 Update**: We’ve built a [free short course](https://bit.ly/4aRnn7Z) that brings the contents of this post up-to-date with animations:

**2025 年更新**：我们制作了一门[免费短课程](https://bit.ly/4aRnn7Z)，用动画把本文的内容更新到最新。

## A High-Level Look ｜ 高层视角

Let’s begin by looking at the model as a single black box. In a machine translation application, it would take a sentence in one language, and output its translation in another.

我们先把这个模型看作一个单独的黑箱。在机器翻译应用中，它接收一种语言的一个句子，输出另一种语言中的译文。

![](https://jalammar.github.io/images/t/the_transformer_3.png)

Popping open that Optimus Prime goodness, we see an encoding component, a decoding component, and connections between them.

把这台"擎天柱"打开，我们会看到一个编码组件、一个解码组件，以及二者之间的连接。

![](https://jalammar.github.io/images/t/The_transformer_encoders_decoders.png)

The encoding component is a stack of encoders (the paper stacks six of them on top of each other – there’s nothing magical about the number six, one can definitely experiment with other arrangements). The decoding component is a stack of decoders of the same number.

编码组件由若干个编码器堆叠而成（论文中叠加了六个——"六"这个数字并没有什么魔力，完全可以尝试其他层数）。解码组件由同样数量的解码器堆叠而成。

![](https://jalammar.github.io/images/t/The_transformer_encoder_decoder_stack.png)

The encoders are all identical in structure (yet they do not share weights). Each one is broken down into two sub-layers:

所有编码器的结构完全相同（但它们并不共享权重）。每个编码器都可以拆成两个子层：

![](https://jalammar.github.io/images/t/Transformer_encoder.png)

The encoder’s inputs first flow through a self-attention layer – a layer that helps the encoder look at other words in the input sentence as it encodes a specific word. We’ll look closer at self-attention later in the post.

编码器的输入首先流经自注意力层——这一层帮助编码器在编码某个特定词时去关注输入句子中的其他词。后文我们会更仔细地观察自注意力。

The outputs of the self-attention layer are fed to a feed-forward neural network. The exact same feed-forward network is independently applied to each position.

自注意力层的输出被送入前馈神经网络。完全相同的前馈网络会被独立地应用到每一个位置。

The decoder has both those layers, but between them is an attention layer that helps the decoder focus on relevant parts of the input sentence (similar what attention does in [seq2seq models](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)).

解码器同样拥有这两个子层，但在它们之间还多了一层注意力层，帮助解码器聚焦于输入句子中相关的部分（类似于注意力在 [seq2seq 模型](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)中所起的作用）。

![](https://jalammar.github.io/images/t/Transformer_decoder.png)

## Bringing The Tensors Into The Picture ｜ 把张量引入视野

Now that we’ve seen the major components of the model, let’s start to look at the various vectors/tensors and how they flow between these components to turn the input of a trained model into an output.

既然已经看过模型的主要组件，接下来我们看看各种向量/张量，以及它们如何在组件之间流动，从而把训练好的模型的输入变成输出。

As is the case in NLP applications in general, we begin by turning each input word into a vector using an [embedding algorithm](https://medium.com/deeper-learning/glossary-of-deep-learning-word-embedding-f90c3cec34ca).

正如一般的 NLP 应用那样，我们首先用[嵌入算法](https://medium.com/deeper-learning/glossary-of-deep-learning-word-embedding-f90c3cec34ca)把每个输入词变成一个向量。

![](https://jalammar.github.io/images/t/embeddings.png)

*Each word is embedded into a vector of size 512. We'll represent those vectors with these simple boxes.*

*每个词都被嵌入成一个大小为 512 的向量。我们用这些简单的方框来表示这些向量。*

The embedding only happens in the bottom-most encoder. The abstraction that is common to all the encoders is that they receive a list of vectors each of the size 512 – In the bottom encoder that would be the word embeddings, but in other encoders, it would be the output of the encoder that’s directly below. The size of this list is hyperparameter we can set – basically it would be the length of the longest sentence in our training dataset.

嵌入只发生在最底层的编码器中。所有编码器共有的抽象是：它们接收一个向量列表，每个向量的维度都是 512——在最底层编码器中，这个列表是词嵌入；而在其他编码器中，它是正下方那个编码器的输出。这个列表的长度是一个我们可以设定的超参数——基本上就等于训练数据集中最长句子的长度。

After embedding the words in our input sequence, each of them flows through each of the two layers of the encoder.

把输入序列中的词嵌入之后，每个词都会依次流过编码器的两个子层。

![](https://jalammar.github.io/images/t/encoder_with_tensors.png)

Here we begin to see one key property of the Transformer, which is that the word in each position flows through its own path in the encoder. There are dependencies between these paths in the self-attention layer. The feed-forward layer does not have those dependencies, however, and thus the various paths can be executed in parallel while flowing through the feed-forward layer.

这里我们开始看到 Transformer 的一个关键特性：每个位置的词在编码器中走属于自己的路径。在自注意力层中，这些路径之间存在依赖关系。但前馈层没有这种依赖，因此在流经前馈层时，各条路径可以并行执行。

Next, we’ll switch up the example to a shorter sentence and we’ll look at what happens in each sub-layer of the encoder.

接下来，我们换一个更短的句子，看看编码器的每个子层内部发生了什么。

## Now We’re Encoding! ｜ 现在开始编码！

As we’ve mentioned already, an encoder receives a list of vectors as input. It processes this list by passing these vectors into a ‘self-attention’ layer, then into a feed-forward neural network, then sends out the output upwards to the next encoder.

如前所述，编码器接收一个向量列表作为输入。它把这些向量送入"自注意力"层，再送入前馈神经网络，然后把输出向上传递给下一个编码器。

![](https://jalammar.github.io/images/t/encoder_with_tensors_2.png)

*The word at each position passes through a self-attention process. Then, they each pass through a feed-forward neural network -- the exact same network with each vector flowing through it separately.*

*每个位置的词都会经过一个自注意力过程。然后它们各自经过一个前馈神经网络——完全相同的网络，每个向量分别流过它。*

## Self-Attention at a High Level ｜ 自注意力：直观理解

Don’t be fooled by me throwing around the word “self-attention” like it’s a concept everyone should be familiar with. I had personally never came across the concept until reading the Attention is All You Need paper. Let us distill how it works.

别因为我把"自注意力"这个词随口抛出，就以为这是人人都该熟悉的概念。在读到《Attention is All You Need》之前，我个人从未接触过这个概念。我们来提炼一下它的工作原理。

Say the following sentence is an input sentence we want to translate:

假设下面这句话是我们要翻译的输入句子：

”`The animal didn't cross the street because it was too tired`”

"这只动物没有穿过街道，因为它太累了。"

What does “it” in this sentence refer to? Is it referring to the street or to the animal? It’s a simple question to a human, but not as simple to an algorithm.

这句话里的"it"指的是什么？是街道，还是动物？对人来说这是个简单问题，但对算法来说就没那么简单了。

When the model is processing the word “it”, self-attention allows it to associate “it” with “animal”.

当模型处理"it"这个词时，自注意力让它把"it"与"animal"关联起来。

As the model processes each word (each position in the input sequence), self attention allows it to look at other positions in the input sequence for clues that can help lead to a better encoding for this word.

当模型处理每个词（输入序列中的每个位置）时，自注意力让它去查看输入序列中的其他位置，寻找有助于更好地编码当前词的线索。

If you’re familiar with RNNs, think of how maintaining a hidden state allows an RNN to incorporate its representation of previous words/vectors it has processed with the current one it’s processing. Self-attention is the method the Transformer uses to bake the “understanding” of other relevant words into the one we’re currently processing.

如果你熟悉 RNN，可以想一想：维护隐状态如何让 RNN 把对已经处理过的词/向量的表示，与当前正在处理的词结合起来。自注意力则是 Transformer 把"对其他相关词的理解"烘焙进当前词表示中的方法。

![](https://jalammar.github.io/images/t/transformer_self-attention_visualization.png)

*As we are encoding the word "it" in encoder #5 (the top encoder in the stack), part of the attention mechanism was focusing on "The Animal", and baked a part of its representation into the encoding of "it".*

*当我们在第 5 个编码器（堆叠中最顶层的编码器）中编码"it"时，注意力机制的一部分正聚焦在"The animal"上，并把其表示的一部分融入了"it"的编码中。*

Be sure to check out the [Tensor2Tensor notebook](https://colab.research.google.com/github/tensorflow/tensor2tensor/blob/master/tensor2tensor/notebooks/hello_t2t.ipynb) where you can load a Transformer model, and examine it using this interactive visualization.

务必看看 [Tensor2Tensor notebook](https://colab.research.google.com/github/tensorflow/tensor2tensor/blob/master/tensor2tensor/notebooks/hello_t2t.ipynb)，你可以在其中加载一个 Transformer 模型，并用这种交互式可视化来检视它。

## Self-Attention in Detail ｜ 自注意力：细节

Let’s first look at how to calculate self-attention using vectors, then proceed to look at how it’s actually implemented – using matrices.

我们先看看如何用向量计算自注意力，然后再看它在实际实现中是如何用矩阵完成的。

The **first step** in calculating self-attention is to create three vectors from each of the encoder’s input vectors (in this case, the embedding of each word). So for each word, we create a Query vector, a Key vector, and a Value vector. These vectors are created by multiplying the embedding by three matrices that we trained during the training process.

计算自注意力的**第一步**，是从编码器的每个输入向量（这里就是每个词的嵌入）生成三个向量。也就是说，对每个词，我们创建一个查询向量 Query、一个键向量 Key 和一个值向量 Value。它们由嵌入乘以三个在训练过程中学到的矩阵得到。

Notice that these new vectors are smaller in dimension than the embedding vector. Their dimensionality is 64, while the embedding and encoder input/output vectors have dimensionality of 512. They don’t HAVE to be smaller, this is an architecture choice to make the computation of multiheaded attention (mostly) constant.

注意，这些新向量的维度比嵌入向量小：它们的维度是 64，而嵌入以及编码器输入/输出向量的维度是 512。它们并非*必须*更小，这只是为了让多头注意力的计算量（基本）保持恒定而做出的架构选择。

![](https://jalammar.github.io/images/t/transformer_self_attention_vectors.png)

*Multiplying x1 by the WQ weight matrix produces q1, the "query" vector associated with that word. We end up creating a "query", a "key", and a "value" projection of each word in the input sentence.*

*用 x1 乘以 WQ 权重矩阵得到 q1，即这个词对应的"查询"向量。最终我们会为输入句子中的每个词都生成一个"查询"、一个"键"和一个"值"投影。*

What are the “query”, “key”, and “value” vectors? They’re abstractions that are useful for calculating and thinking about attention. Once you proceed with reading how attention is calculated below, you’ll know pretty much all you need to know about the role each of these vectors plays.

"查询""键""值"向量究竟是什么？它们是为了便于计算和思考注意力而引入的抽象。读完下文关于注意力如何计算的内容，你基本就会了解这三个向量各自扮演的角色。

The **second step** in calculating self-attention is to calculate a score. Say we’re calculating the self-attention for the first word in this example, “Thinking”. We need to score each word of the input sentence against this word. The score determines how much focus to place on other parts of the input sentence as we encode a word at a certain position.

计算自注意力的**第二步**是计算分数。假设我们在为例子中的第一个词"Thinking"计算自注意力。我们需要用这个词去给输入句子中的每个词打分。这个分数决定了在编码某个位置的词时，应当把多少注意力放在输入句子的其他部分上。

The score is calculated by taking the dot product of the query vector with the key vector of the respective word we’re scoring. So if we’re processing the self-attention for the word in position #1, the first score would be the dot product of q1 and k1. The second score would be the dot product of q1 and k2.

分数的计算方式是：用查询向量与我们要打分的那个词的键向量做点积。所以，如果我们在计算位置 #1 的词的自注意力，第一个分数就是 q1 与 k1 的点积，第二个分数则是 q1 与 k2 的点积。

![](https://jalammar.github.io/images/t/transformer_self_attention_score.png)

The **third and fourth steps** are to divide the scores by 8 (the square root of the dimension of the key vectors used in the paper – 64. This leads to having more stable gradients. There could be other possible values here, but this is the default), then pass the result through a softmax operation. Softmax normalizes the scores so they’re all positive and add up to 1.

**第三步和第四步**是把分数除以 8（论文中所用键向量维度 64 的平方根，这样梯度会更稳定；这里也可以取其他值，但这是默认做法），然后把结果送入 softmax 运算。Softmax 会对分数做归一化，使它们全部为正且相加等于 1。

![](https://jalammar.github.io/images/t/self-attention_softmax.png)

This softmax score determines how much each word will be expressed at this position. Clearly the word at this position will have the highest softmax score, but sometimes it’s useful to attend to another word that is relevant to the current word.

这个 softmax 分数决定了每个词在这个位置上会被表达多少。显然，当前位置的词会得到最高的 softmax 分数，但有时关注另一个与当前词相关的词也是有用的。

The **fifth step** is to multiply each value vector by the softmax score (in preparation to sum them up). The intuition here is to keep intact the values of the word(s) we want to focus on, and drown-out irrelevant words (by multiplying them by tiny numbers like 0.001, for example).

**第五步**是把每个值向量乘以对应的 softmax 分数（为后面求和做准备）。这里的直觉是：完整保留我们想关注的那个（些）词的数值，同时把无关词"淹没"掉（例如乘以 0.001 这样很小的数）。

The **sixth step** is to sum up the weighted value vectors. This produces the output of the self-attention layer at this position (for the first word).

**第六步**是把加权后的值向量相加。这样得到的，就是该位置（对第一个词而言）自注意力层的输出。

![](https://jalammar.github.io/images/t/self-attention-output.png)

That concludes the self-attention calculation. The resulting vector is one we can send along to the feed-forward neural network. In the actual implementation, however, this calculation is done in matrix form for faster processing. So let’s look at that now that we’ve seen the intuition of the calculation on the word level.

自注意力的计算到此结束。得到的向量可以送入前馈神经网络。不过在实际实现中，为了处理更快，这一计算是以矩阵形式完成的。既然我们已经在词级别看清了计算的直觉，现在就来看矩阵形式。

## Matrix Calculation of Self-Attention ｜ 自注意力的矩阵计算

**The first step** is to calculate the Query, Key, and Value matrices. We do that by packing our embeddings into a matrix X, and multiplying it by the weight matrices we’ve trained (WQ, WK, WV).

**第一步**是计算 Query、Key、Value 三个矩阵。做法是把嵌入打包成矩阵 X，然后乘以我们训练得到的权重矩阵（WQ、WK、WV）。

![](https://jalammar.github.io/images/t/self-attention-matrix-calculation.png)

*Every row in the X matrix corresponds to a word in the input sentence. We again see the difference in size of the embedding vector (512, or 4 boxes in the figure), and the q/k/v vectors (64, or 3 boxes in the figure)*

*X 矩阵中的每一行都对应输入句子中的一个词。我们再次看到嵌入向量（512 维，图中画成 4 个方框）与 q/k/v 向量（64 维，图中画成 3 个方框）在尺寸上的差别。*

**Finally**, since we’re dealing with matrices, we can condense steps two through six in one formula to calculate the outputs of the self-attention layer.

**最后**，由于处理的是矩阵，我们可以把第二到第六步压缩成一个公式，一次性算出整个自注意力层的输出。

![](https://jalammar.github.io/images/t/self-attention-matrix-calculation-2.png)

*The self-attention calculation in matrix form*

*矩阵形式的自注意力计算。*

## The Beast With Many Heads ｜ 多头注意力

The paper further refined the self-attention layer by adding a mechanism called “multi-headed” attention. This improves the performance of the attention layer in two ways:

论文进一步改进了自注意力层，加入了一种名为"多头（multi-headed）"注意力的机制。它从两个方面提升了注意力层的表现：

**1.** It expands the model’s ability to focus on different positions. Yes, in the example above, z1 contains a little bit of every other encoding, but it could be dominated by the actual word itself. If we’re translating a sentence like “The animal didn’t cross the street because it was too tired”, it would be useful to know which word “it” refers to.

**1.** 它扩展了模型同时关注不同位置的能力。确实，在上面的例子里，z1 包含了一点其他所有编码的信息，但它可能被当前词本身所主导。如果我们要翻译"The animal didn't cross the street because it was too tired"这样的句子，知道"it"指的是哪个词会很有用。

**2.** It gives the attention layer multiple “representation subspaces”. As we’ll see next, with multi-headed attention we have not only one, but multiple sets of Query/Key/Value weight matrices (the Transformer uses eight attention heads, so we end up with eight sets for each encoder/decoder). Each of these sets is randomly initialized. Then, after training, each set is used to project the input embeddings (or vectors from lower encoders/decoders) into a different representation subspace.

**2.** 它为注意力层提供了多个"表示子空间"。接下来我们会看到，在多头注意力下，我们不止有一组，而是有多组 Query/Key/Value 权重矩阵（Transformer 使用八个注意力头，因此每个编码器/解码器最终有八组）。每一组都是随机初始化的。训练之后，每一组都会把输入嵌入（或来自更低层编码器/解码器的向量）投影到一个不同的表示子空间。

![](https://jalammar.github.io/images/t/transformer_attention_heads_qkv.png)

*With multi-headed attention, we maintain separate Q/K/V weight matrices for each head resulting in different Q/K/V matrices. As we did before, we multiply X by the WQ/WK/WV matrices to produce Q/K/V matrices.*

*在多头注意力中，我们为每个头维护各自独立的 Q/K/V 权重矩阵，从而得到不同的 Q/K/V 矩阵。和之前一样，我们用 X 乘以 WQ/WK/WV 矩阵来生成 Q/K/V 矩阵。*

If we do the same self-attention calculation we outlined above, just eight different times with different weight matrices, we end up with eight different Z matrices

如果我们把上面描述的自注意力计算用不同的权重矩阵做八次，就会得到八个不同的 Z 矩阵。

![](https://jalammar.github.io/images/t/transformer_attention_heads_z.png)

This leaves us with a bit of a challenge. The feed-forward layer is not expecting eight matrices – it’s expecting a single matrix (a vector for each word). So we need a way to condense these eight down into a single matrix.

这给我们出了个小难题：前馈层并不期待八个矩阵，它期待的是单个矩阵（每个词一个向量）。所以我们需要一种方法，把这八个矩阵压缩成一个。

How do we do that? We concat the matrices then multiply them by an additional weights matrix WO.

怎么做呢？我们把矩阵拼接（concat）起来，再乘以一个额外的权重矩阵 WO。

![](https://jalammar.github.io/images/t/transformer_attention_heads_weight_matrix_o.png)

That’s pretty much all there is to multi-headed self-attention. It’s quite a handful of matrices, I realize. Let me try to put them all in one visual so we can look at them in one place

多头自注意力基本上就是这些。我知道这里涉及的矩阵确实有点多，让我把它们放到同一张图里，方便一次看全。

![](https://jalammar.github.io/images/t/transformer_multi-headed_self-attention-recap.png)

Now that we have touched upon attention heads, let’s revisit our example from before to see where the different attention heads are focusing as we encode the word “it” in our example sentence:

既然已经谈过注意力头，我们回到之前的例子，看看在编码例句中的"it"时，不同的注意力头分别关注哪里：

![](https://jalammar.github.io/images/t/transformer_self-attention_visualization_2.png)

*As we encode the word "it", one attention head is focusing most on "the animal", while another is focusing on "tired" -- in a sense, the model's representation of the word "it" bakes in some of the representation of both "animal" and "tired".*

*当我们在编码"it"这个词时，一个注意力头最关注"the animal"，而另一个头则关注"tired"——某种意义上，模型对"it"这个词的表示融入了"animal"和"tired"两者的部分表示。*

If we add all the attention heads to the picture, however, things can be harder to interpret:

不过，如果把这所有的注意力头都画进图里，事情反而变得更难解读了：

![](https://jalammar.github.io/images/t/transformer_self-attention_visualization_3.png)

## Representing The Order of The Sequence Using Positional Encoding ｜ 用位置编码表示序列顺序

One thing that’s missing from the model as we have described it so far is a way to account for the order of the words in the input sequence.

到目前为止所描述的模型中，还缺少一样东西：如何体现输入序列中词序的信息。

To address this, the transformer adds a vector to each input embedding. These vectors follow a specific pattern that the model learns, which helps it determine the position of each word, or the distance between different words in the sequence. The intuition here is that adding these values to the embeddings provides meaningful distances between the embedding vectors once they’re projected into Q/K/V vectors and during dot-product attention.

为了解决这个问题，Transformer 给每个输入嵌入加上一个向量。这些向量遵循模型学到的某种特定模式，帮助模型确定每个词的位置，或者序列中不同词之间的距离。这里的直觉是：把这些值加到嵌入上之后，当嵌入被投影成 Q/K/V 向量并做点积注意力时，向量之间就能体现出有意义的距离。

![](https://jalammar.github.io/images/t/transformer_positional_encoding_vectors.png)

*To give the model a sense of the order of the words, we add positional encoding vectors -- the values of which follow a specific pattern.*

*为了让模型感知词的顺序，我们加上位置编码向量——它们的取值遵循某种特定的模式。*

If we assumed the embedding has a dimensionality of 4, the actual positional encodings would look like this:

如果我们假设嵌入的维度是 4，真实的位置编码看起来会是这样：

![](https://jalammar.github.io/images/t/transformer_positional_encoding_example.png)

*A real example of positional encoding with a toy embedding size of 4*

*嵌入维度取 4 这一玩具尺寸时的真实位置编码示例。*

What might this pattern look like?

这个模式可能长什么样呢？

In the following figure, each row corresponds to a positional encoding of a vector. So the first row would be the vector we’d add to the embedding of the first word in an input sequence. Each row contains 512 values – each with a value between 1 and -1. We’ve color-coded them so the pattern is visible.

在下图中，每一行对应一个向量的位置编码。因此第一行就是要加到输入序列第一个词嵌入上的向量。每一行包含 512 个值——每个值都介于 1 和 -1 之间。我们对它们做了着色，以便让这个模式清晰可见。

![](https://jalammar.github.io/images/t/transformer_positional_encoding_large_example.png)

*A real example of positional encoding for 20 words (rows) with an embedding size of 512 (columns). You can see that it appears split in half down the center. That's because the values of the left half are generated by one function (which uses sine), and the right half is generated by another function (which uses cosine). They're then concatenated to form each of the positional encoding vectors.*

*20 个词（行）、嵌入维度为 512（列）时的真实位置编码示例。你可以看到它似乎在正中间被分成了两半：这是因为左半部分的值由一个函数生成（使用正弦），右半部分的值由另一个函数生成（使用余弦），然后再把它们拼接起来，构成每一个位置编码向量。*

The formula for positional encoding is described in the paper (section 3.5). You can see the code for generating positional encodings in [get_timing_signal_1d()](https://github.com/tensorflow/tensor2tensor/blob/23bd23b9830059fbc349381b70d9429b5c40a139/tensor2tensor/layers/common_attention.py). This is not the only possible method for positional encoding. It, however, gives the advantage of being able to scale to unseen lengths of sequences (e.g. if our trained model is asked to translate a sentence longer than any of those in our training set).

位置编码的公式在论文第 3.5 节中有描述。生成位置编码的代码可以在 [get_timing_signal_1d()](https://github.com/tensorflow/tensor2tensor/blob/23bd23b9830059fbc349381b70d9429b5c40a139/tensor2tensor/layers/common_attention.py) 中看到。这并不是唯一可能的位置编码方法，但它的优势在于能够扩展到训练时未见过的序列长度（例如，当我们训练好的模型被要求翻译一个比训练集中任何句子都长的句子时）。

**July 2020 Update:** The positional encoding shown above is from the Tensor2Tensor implementation of the Transformer. The method shown in the paper is slightly different in that it doesn’t directly concatenate, but interweaves the two signals. The following figure shows what that looks like. [Here’s the code to generate it](https://github.com/jalammar/jalammar.github.io/blob/master/notebookes/transformer/transformer_positional_encoding_graph.ipynb):

**2020 年 7 月更新：** 上面展示的位置编码来自 Transformer 的 Tensor2Tensor 实现。论文中展示的方法略有不同：它不是直接拼接，而是把两个信号交织在一起。下图展示了这种做法的效果。[这里是生成它的代码](https://github.com/jalammar/jalammar.github.io/blob/master/notebookes/transformer/transformer_positional_encoding_graph.ipynb)：

![](https://jalammar.github.io/images/t/attention-is-all-you-need-positional-encoding.png)

## The Residuals ｜ 残差连接

One detail in the architecture of the encoder that we need to mention before moving on, is that each sub-layer (self-attention, ffnn) in each encoder has a residual connection around it, and is followed by a [layer-normalization](https://arxiv.org/abs/1607.06450) step.

在继续之前，还有一个编码器架构中的细节需要提及：每个编码器中的每个子层（自注意力、前馈网络）外面都有一条残差连接，并且后面紧跟一个[层归一化（layer-normalization）](https://arxiv.org/abs/1607.06450)步骤。

![](https://jalammar.github.io/images/t/transformer_resideual_layer_norm.png)

If we’re to visualize the vectors and the layer-norm operation associated with self attention, it would look like this:

如果我们把与自注意力相关的向量和层归一化操作可视化，大概会是这样：

![](https://jalammar.github.io/images/t/transformer_resideual_layer_norm_2.png)

This goes for the sub-layers of the decoder as well. If we’re to think of a Transformer of 2 stacked encoders and decoders, it would look something like this:

解码器的各个子层也是如此。如果我们想象一个堆叠了两层编码器和两层解码器的 Transformer，它大概长这样：

![](https://jalammar.github.io/images/t/transformer_resideual_layer_norm_3.png)

## The Decoder Side ｜ 解码器一侧

Now that we’ve covered most of the concepts on the encoder side, we basically know how the components of decoders work as well. But let’s take a look at how they work together.

现在我们已经覆盖了编码器一侧的大部分概念，解码器各组件的工作原理也就基本清楚了。不过，我们还是来看一看它们是如何协同工作的。

The encoder start by processing the input sequence. The output of the top encoder is then transformed into a set of attention vectors K and V. These are to be used by each decoder in its “encoder-decoder attention” layer which helps the decoder focus on appropriate places in the input sequence:

编码器先处理输入序列。顶层编码器的输出随后被转换成一组注意力向量 K 和 V。这些向量会供每个解码器的"编码器-解码器注意力"层使用，帮助解码器聚焦于输入序列中合适的位置：

![](https://jalammar.github.io/images/t/transformer_decoding_1.gif)

*After finishing the encoding phase, we begin the decoding phase. Each step in the decoding phase outputs an element from the output sequence (the English translation sentence in this case).*

*编码阶段结束后，我们开始解码阶段。解码阶段的每一步都会输出输出序列中的一个元素（在本例中是英文译文句子中的一个词）。*

The following steps repeat the process until a special symbol is reached indicating the transformer decoder has completed its output. The output of each step is fed to the bottom decoder in the next time step, and the decoders bubble up their decoding results just like the encoders did. And just like we did with the encoder inputs, we embed and add positional encoding to those decoder inputs to indicate the position of each word.

后续步骤会重复这一过程，直到遇到一个表示 Transformer 解码器已完成输出的特殊符号。每一步的输出都会送入下一时间步的底层解码器，解码器再像编码器那样把解码结果逐层向上传递。和编码器输入一样，我们也会对这些解码器输入做嵌入并加上位置编码，以标明每个词的位置。

![](https://jalammar.github.io/images/t/transformer_decoding_2.gif)

The self attention layers in the decoder operate in a slightly different way than the one in the encoder:

解码器中的自注意力层与编码器中的运作方式略有不同：

In the decoder, the self-attention layer is only allowed to attend to earlier positions in the output sequence. This is done by masking future positions (setting them to `-inf`) before the softmax step in the self-attention calculation.

在解码器中，自注意力层只被允许关注输出序列中靠前的位置。实现方式是在自注意力计算中做 softmax 之前，对未来位置做掩码（把它们设为 `-inf`）。

The “Encoder-Decoder Attention” layer works just like multiheaded self-attention, except it creates its Queries matrix from the layer below it, and takes the Keys and Values matrix from the output of the encoder stack.

"编码器-解码器注意力"层的工作方式与多头自注意力完全一样，区别在于它的 Query 矩阵来自它下面那一层，而 Key 和 Value 矩阵则取自编码器堆叠的输出。

## The Final Linear and Softmax Layer ｜ 最后的线性层与 Softmax 层

The decoder stack outputs a vector of floats. How do we turn that into a word? That’s the job of the final Linear layer which is followed by a Softmax Layer.

解码器堆叠输出的是一个浮点数向量。我们如何把它变成一个词？这就是最后的线性层的任务，它后面还跟着一个 Softmax 层。

The Linear layer is a simple fully connected neural network that projects the vector produced by the stack of decoders, into a much, much larger vector called a logits vector.

线性层就是一个简单的全连接神经网络，它把解码器堆叠产生的向量投影成一个维度大得多的向量，称为 logits 向量。

Let’s assume that our model knows 10,000 unique English words (our model’s “output vocabulary”) that it’s learned from its training dataset. This would make the logits vector 10,000 cells wide – each cell corresponding to the score of a unique word. That is how we interpret the output of the model followed by the Linear layer.

假设我们的模型认识 10,000 个不同的英文单词（这就是模型的"输出词表"，是它从训练数据集中学到的）。那么 logits 向量就有 10,000 个单元宽——每个单元对应一个独特单词的分数。这就是我们如何解释模型经过线性层之后的输出。

The softmax layer then turns those scores into probabilities (all positive, all add up to 1.0). The cell with the highest probability is chosen, and the word associated with it is produced as the output for this time step.

接着，softmax 层把这些分数转换成概率（全部为正，且总和为 1.0）。概率最高的那个单元会被选中，与之关联的词就作为这一时间步的输出。

![](https://jalammar.github.io/images/t/transformer_decoder_output_softmax.png)

*This figure starts from the bottom with the vector produced as the output of the decoder stack. It is then turned into an output word.*

*这张图从底部开始，是解码器堆叠输出的向量。随后它被转换成一个输出词。*

## Recap Of Training ｜ 训练过程回顾

Now that we’ve covered the entire forward-pass process through a trained Transformer, it would be useful to glance at the intuition of training the model.

现在我们已经完整走过了训练好的 Transformer 的前向传播过程，接下来回顾一下训练模型的直觉会很有帮助。

During training, an untrained model would go through the exact same forward pass. But since we are training it on a labeled training dataset, we can compare its output with the actual correct output.

训练时，一个未经训练的模型也会走完全相同的前向传播。但由于我们是在有标注的训练数据集上训练它，因此可以把它的输出与真正正确的输出做比较。

To visualize this, let’s assume our output vocabulary only contains six words(“a”, “am”, “i”, “thanks”, “student”, and “<eos>” (short for ‘end of sentence’)).

为了把这一点可视化，假设我们的输出词表只包含六个词（"a""am""i""thanks""student" 和 "<eos>"，即"句子结束"的缩写）。

![](https://jalammar.github.io/images/t/vocabulary.png)

*The output vocabulary of our model is created in the preprocessing phase before we even begin training.*

*我们模型的输出词表是在真正开始训练之前的预处理阶段创建的。*

Once we define our output vocabulary, we can use a vector of the same width to indicate each word in our vocabulary. This also known as one-hot encoding. So for example, we can indicate the word “am” using the following vector:

定义好输出词表之后，我们就可以用一个同样宽度的向量来表示词表中的每个词，这也称为独热编码（one-hot encoding）。例如，我们可以用下面这个向量表示"am"这个词：

![](https://jalammar.github.io/images/t/one-hot-vocabulary-example.png)

*Example: one-hot encoding of our output vocabulary*

*示例：输出词表的独热编码。*

Following this recap, let’s discuss the model’s loss function – the metric we are optimizing during the training phase to lead up to a trained and hopefully amazingly accurate model.

回顾完之后，我们来讨论模型的损失函数——也就是训练阶段我们要优化的指标，目标是得到一个训练好的、并且希望非常准确的模型。

## The Loss Function ｜ 损失函数

Say we are training our model. Say it’s our first step in the training phase, and we’re training it on a simple example – translating “merci” into “thanks”.

假设我们正在训练模型。假设这是训练阶段的第一步，我们用一个简单的例子来训练——把"merci"翻译成"thanks"。

What this means, is that we want the output to be a probability distribution indicating the word “thanks”. But since this model is not yet trained, that’s unlikely to happen just yet.

这意味着，我们希望输出是一个表示"thanks"这个词的概率分布。但由于这个模型还没有训练，眼下这还不太可能发生。

![](https://jalammar.github.io/images/t/transformer_logits_output_and_label.png)

*Since the model's parameters (weights) are all initialized randomly, the (untrained) model produces a probability distribution with arbitrary values for each cell/word. We can compare it with the actual output, then tweak all the model's weights using backpropagation to make the output closer to the desired output.*

*由于模型的参数（权重）都是随机初始化的，未训练的模型会为每个单元/单词产生一个取值任意的概率分布。我们可以把它与实际输出相比较，然后用反向传播调整模型的所有权重，让输出更接近期望的输出。*

How do you compare two probability distributions? We simply subtract one from the other. For more details, look at [cross-entropy](https://colah.github.io/posts/2015-09-Visual-Information/) and [Kullback–Leibler divergence](https://www.countbayesie.com/blog/2017/5/9/kullback-leibler-divergence-explained).

如何比较两个概率分布？简单地把它们相减即可。更多细节可以参考[交叉熵](https://colah.github.io/posts/2015-09-Visual-Information/)和 [KL 散度](https://www.countbayesie.com/blog/2017/5/9/kullback-leibler-divergence-explained)。

But note that this is an oversimplified example. More realistically, we’ll use a sentence longer than one word. For example – input: “je suis étudiant” and expected output: “i am a student”. What this really means, is that we want our model to successively output probability distributions where:

但请注意，这是一个过度简化的例子。更现实的情况是，我们会使用比一个词更长的句子。例如输入"je suis étudiant"，期望输出"i am a student"。这真正意味着，我们希望模型依次输出这样的概率分布：

- Each probability distribution is represented by a vector of width vocab_size (6 in our toy example, but more realistically a number like 30,000 or 50,000)

- 每个概率分布都由一个宽度为 vocab_size 的向量表示（在我们的玩具例子中是 6，现实中更可能是 30,000 或 50,000 这样的数字）。

- The first probability distribution has the highest probability at the cell associated with the word “i”

- 第一个概率分布在"i"所对应的单元上概率最高。

- The second probability distribution has the highest probability at the cell associated with the word “am”

- 第二个概率分布在"am"所对应的单元上概率最高。

- And so on, until the fifth output distribution indicates ‘`<end of sentence>`’ symbol, which also has a cell associated with it from the 10,000 element vocabulary.

- 依此类推，直到第五个输出分布指示 '`<end of sentence>`' 符号，而这个符号在这个 10,000 元素的词表中同样有对应的单元。

![](https://jalammar.github.io/images/t/output_target_probability_distributions.png)

*The targeted probability distributions we'll train our model against in the training example for one sample sentence.*

*在单词句子的训练示例中，我们要用来训练模型的目标概率分布。*

After training the model for enough time on a large enough dataset, we would hope the produced probability distributions would look like this:

在足够大的数据集上训练足够长的时间之后，我们希望产生的概率分布会是这样：

![](https://jalammar.github.io/images/t/output_trained_model_probability_distributions.png)

*Hopefully upon training, the model would output the right translation we expect. Of course it's no real indication if this phrase was part of the training dataset (see: [cross validation](https://www.youtube.com/watch?v=TIgfjmp-4BA)). Notice that every position gets a little bit of probability even if it's unlikely to be the output of that time step -- that's a very useful property of softmax which helps the training process.*

*希望训练之后模型能够输出我们期望的正确译文。当然，如果这个短语本来就在训练数据集中，那这并不能说明什么（参见[交叉验证](https://www.youtube.com/watch?v=TIgfjmp-4BA)）。注意每个位置都分到了少量概率，即使它不太可能是该时间步的输出——这是 softmax 一个非常有用的特性，对训练过程很有帮助。*

Now, because the model produces the outputs one at a time, we can assume that the model is selecting the word with the highest probability from that probability distribution and throwing away the rest. That’s one way to do it (called greedy decoding). Another way to do it would be to hold on to, say, the top two words (say, ‘I’ and ‘a’ for example), then in the next step, run the model twice: once assuming the first output position was the word ‘I’, and another time assuming the first output position was the word ‘a’, and whichever version produced less error considering both positions #1 and #2 is kept. We repeat this for positions #2 and #3…etc. This method is called “beam search”, where in our example, beam_size was two (meaning that at all times, two partial hypotheses (unfinished translations) are kept in memory), and top_beams is also two (meaning we’ll return two translations). These are both hyperparameters that you can experiment with.

现在，由于模型是一次产生一个输出的，我们可以假定模型会从该概率分布中挑出概率最高的词，并把其余的都丢掉。这是一种做法（称为贪心解码，greedy decoding）。另一种做法是保留概率最高的两个词（比如"I"和"a"），然后在下一步把模型运行两次：一次假设第一个输出位置是"I"，一次假设是"a"，然后保留在位置 #1 和 #2 上总误差更小的那个版本。我们对位置 #2 和 #3 重复同样的操作……以此类推。这种方法叫做"束搜索"（beam search）。在我们的例子里，beam_size 为 2（意味着内存中始终保留两个部分假设，也就是尚未完成的译文），top_beams 也为 2（意味着最终返回两个译文）。这两者都是你可以自行尝试的超参数。

## Go Forth And Transform ｜ 出发吧，去 Transform

I hope you’ve found this a useful place to start to break the ice with the major concepts of the Transformer. If you want to go deeper, I’d suggest these next steps:

希望这里能成为你入门 Transformer 核心概念的一个有用起点。如果你想更深入，我建议接下来的步骤：

- Read the [Attention Is All You Need](https://arxiv.org/abs/1706.03762) paper, the Transformer blog post ([Transformer: A Novel Neural Network Architecture for Language Understanding](https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html)), and the [Tensor2Tensor announcement](https://ai.googleblog.com/2017/06/accelerating-deep-learning-research.html).

- 阅读 [Attention Is All You Need](https://arxiv.org/abs/1706.03762) 论文、Transformer 博客文章（[Transformer: A Novel Neural Network Architecture for Language Understanding](https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html)）以及 [Tensor2Tensor 发布公告](https://ai.googleblog.com/2017/06/accelerating-deep-learning-research.html)。

- Watch [Łukasz Kaiser’s talk](https://www.youtube.com/watch?v=rBCqOTEfxvg) walking through the model and its details

- 观看 [Łukasz Kaiser 的演讲](https://www.youtube.com/watch?v=rBCqOTEfxvg)，他完整讲解了该模型及其细节。

- Play with the [Jupyter Notebook provided as part of the Tensor2Tensor repo](https://colab.research.google.com/github/tensorflow/tensor2tensor/blob/master/tensor2tensor/notebooks/hello_t2t.ipynb)

- 动手玩一玩 [Tensor2Tensor 仓库提供的 Jupyter Notebook](https://colab.research.google.com/github/tensorflow/tensor2tensor/blob/master/tensor2tensor/notebooks/hello_t2t.ipynb)。

- Explore the [Tensor2Tensor repo](https://github.com/tensorflow/tensor2tensor).

- 探索 [Tensor2Tensor 仓库](https://github.com/tensorflow/tensor2tensor)。

Follow-up works:

后续工作（Follow-up works）：

- [Depthwise Separable Convolutions for Neural Machine Translation](https://arxiv.org/abs/1706.03059) ／ （用于神经机器翻译的深度可分离卷积）

- [One Model To Learn Them All](https://arxiv.org/abs/1706.05137) ／ （一个模型学会所有任务）

- [Discrete Autoencoders for Sequence Models](https://arxiv.org/abs/1801.09797) ／ （用于序列模型的离散自编码器）

- [Generating Wikipedia by Summarizing Long Sequences](https://arxiv.org/abs/1801.10198) ／ （通过摘要化长序列生成维基百科）

- [Image Transformer](https://arxiv.org/abs/1802.05751) ／ （图像 Transformer）

- [Training Tips for the Transformer Model](https://arxiv.org/abs/1804.00247) ／ （Transformer 模型的训练技巧）

- [Self-Attention with Relative Position Representations](https://arxiv.org/abs/1803.02155) ／ （带相对位置表示的自注意力）

- [Fast Decoding in Sequence Models using Discrete Latent Variables](https://arxiv.org/abs/1803.03382) ／ （使用离散潜变量在序列模型中快速解码）

- [Adafactor: Adaptive Learning Rates with Sublinear Memory Cost](https://arxiv.org/abs/1804.04235) ／ （Adafactor：具有次线性内存开销的自适应学习率）

## Acknowledgements ｜ 致谢

Thanks to [Illia Polosukhin](https://twitter.com/ilblackdragon), [Jakob Uszkoreit](http://jakob.uszkoreit.net/), [Llion Jones](https://www.linkedin.com/in/llion-jones-9ab3064b), [Lukasz Kaiser](https://ai.google/research/people/LukaszKaiser), [Niki Parmar](https://twitter.com/nikiparmar09), and [Noam Shazeer](https://dblp.org/pers/hd/s/Shazeer:Noam) for providing feedback on earlier versions of this post.

感谢 [Illia Polosukhin](https://twitter.com/ilblackdragon)、[Jakob Uszkoreit](http://jakob.uszkoreit.net/)、[Llion Jones](https://www.linkedin.com/in/llion-jones-9ab3064b)、[Lukasz Kaiser](https://ai.google/research/people/LukaszKaiser)、[Niki Parmar](https://twitter.com/nikiparmar09) 和 [Noam Shazeer](https://dblp.org/pers/hd/s/Shazeer:Noam) 对本文早期版本提出的反馈。

Please hit me up on [Twitter](https://twitter.com/JayAlammar) for any corrections or feedback.

如有任何纠正或反馈，欢迎在 [Twitter](https://twitter.com/JayAlammar) 上联系我。
