# demixed Shared Component Analysis (dSCA)

![Example Results](https://cdn.rawgit.com/tensorflow/magenta/master/magenta/models/sketch_rnn/assets/sketch_rnn_examples.svg)
*Examples of vector images produced by this generative model.*
This repo contains the TensorFlow code for `sketch-rnn`, the recurrent neural network model described in [Teaching Machines to Draw](https://research.googleblog.com/2017/04/teaching-machines-to-draw.html) and [A Neural Representation of Sketch Drawings](https://arxiv.org/abs/1704.03477).

> Y Takagi<sup>*</sup>, SW Kennerley, J Hirayama<sup>+</sup>, LT Hunt<sup>+</sup><br>
> **Demixed shared component analysis of neuralpopulation data from multiple brain areas**<br>
> NeurIPS 2020, selected as spotlight presentation<br>
> (arXiv link: https://arxiv.org/abs/2006.10212)

MIT license. Contributions welcome.

# Overview of dSCA

dSCA decomposes population activity into a few components, such that the shared components capture the maximum amount of shared information across brain regions while also depending on relevant task parameters. This yields interpretable components that express which variables are shared between different brain regions and when this information is shared across time. 

# Example Usage

Running `simulation.m` provides a simulation results and plotting the results.

### Support

Email yutakagi322@gmail.com with any questions.
