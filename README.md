## From the Tensor to Stable Diffusion

Inspired by [From the Transistor][0].

Machine learning is hard, a lot of tutorials are hard to follow, and
it's hard to understand [software 2.0][1] from first principles.

> You wanna be an ML engineer? Well, here's the steps to get good at that:
>
> 1. Download a paper
> 2. Implement it
> 3. Keep doing this until you have skills
>
> -- *[George Hotz][2]*

#### Section 1: Intro: Cheating our way past the Tensor -- 1 week

- So about those Tensors -- Course overview. Describe how Deep Learning models are buildable using Tensors, and how different architectures like CNNs and RNNs use Tensors in different ways. Understand the concept of backpropagation and gradient descent.
[[video](https://www.youtube.com/watch?v=VMj-3S1tku0)]

#### Section 2: Deep Learning: What is deep learning anyway? -- 1 week

- Building a simple Neural Network -- Your first little program! Getting the model working and learning the basics of deep learning.
[[code](https://github.com/jla524/fromthetensor/blob/main/examples/mnist_from_scratch.ipynb)]
[[video](https://www.youtube.com/watch?v=JRlyw6LO5qo)]

- Building a simple CNN -- An intro chapter to deep learning, learn how to build a simple CNN and understand the concepts of convolution and pooling.
[[code](https://github.com/jla524/fromthetensor/blob/main/examples/cnn.ipynb)]
[[video](https://www.youtube.com/watch?v=KuXjwB4LzSA)]

- Building a simple RNN -- Learn the basics of Recurrent Neural Networks and understand the concept of "memory" that helps them store states of previous inputs.
[[code](https://github.com/jla524/fromthetensor/blob/main/examples/rnn.ipynb)]
[[video](https://www.youtube.com/watch?v=WCUNPb-5EYI)]

#### Section 3: Implementing Papers (Part 1): Vision models -- 3 weeks

- Implementing LeNet -- Learn about the LeNet architecture and its application.
[[code](https://github.com/jla524/fromthetensor/blob/main/examples/lenet.ipynb)]
[[paper](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)]

- Implementing AlexNet -- Learn how to implement AlexNet for image classification tasks.
[[code](https://github.com/jla524/fromthetensor/blob/main/examples/alexnet.ipynb)]
[[paper](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)]

- Implementing ResNet -- Learn how to implement ResNet for image classification tasks.
[[code](https://github.com/jla524/fromthetensor/blob/main/examples/resnet.ipynb)]
[[paper](https://arxiv.org/abs/1512.03385)]

- Building a DCGAN -- Learn how to build a DCGAN and the concept of adversarial training.
[[code](https://github.com/jla524/fromthetensor/blob/main/examples/dcgan.ipynb)]
[[paper](https://arxiv.org/abs/1511.06434)]

#### Section 4: Implementing Papers (Part 2): Language models -- 3 weeks

- Implementing GRU and LSTM -- Learn about the concepts of LSTM and GRU cells.
[[code](https://github.com/jla524/fromthetensor/blob/main/examples/gru_lstm.ipynb)]
[[paper](https://arxiv.org/abs/1412.3555)]

- Implementing CBOW and Skip-Gram -- Learn about the word2vec architecture and its application.
[[code](https://github.com/jla524/fromthetensor/blob/main/examples/cbow_skipgram.ipynb)]
[[paper](https://arxiv.org/abs/1301.3781)]

- Building a Transformer -- Learn about the transformer architecture and its application.
[[code](https://github.com/jla524/fromthetensor/blob/main/examples/transformer.ipynb)]
[[paper](https://arxiv.org/abs/1706.03762)]

- Fine-tuning a BERT -- Learn about the BERT architecture and fine-tuning a pre-trained model.
[[code](https://github.com/jla524/fromthetensor/blob/main/examples/bert.ipynb)]
[[paper](https://arxiv.org/abs/1810.04805)]

#### Section 5: Implementing Papers (Part 3): Vision-Language models -- 1 week

- Building a Stable Diffusion model -- Learn about the Stable Diffusion architecture and its application in image generation tasks.
[[code](https://github.com/jla524/fromthetensor/blob/main/examples/stable_diffusion.ipynb)]
[[paper](https://arxiv.org/abs/2112.10752)]

### Beyond the Tensor

> when ppl ask me how to get better at being a ML engineer i tell them to stop learning about ML and start learning about systems
>
> -- [@yoobinray][4]

See [ideas.md][5] for some ideas.

[0]: https://github.com/geohot/fromthetransistor
[1]: https://karpathy.medium.com/software-2-0-a64152b37c35
[2]: https://youtu.be/N2bXEUSAiTI?t=1315
[3]: https://colab.research.google.com
[4]: https://x.com/yoobinray/status/1965595263290200189
[5]: https://github.com/jla524/fromthetensor/blob/main/ideas.md
