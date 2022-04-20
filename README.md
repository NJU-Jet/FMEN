# FMEN
Lowest memory consumption and second shortest runtime in NTIRE 2022 on Efficient Super-Resolution.

Our paper: [Fast and Memory-Efficient Network Towards Efficient Image Super-Resolution ](https://arxiv.org/abs/2204.08397).

# Main Contribution
1. Enhanced Residual Block.
2. High-Frequency Attention Block.
3. Batch Normalization layers can be applied to attention branch to boost performance.

# Train
Our goal is to design a strightforward but powerful backbone for lightweight image super-resolution, so the testing model is really simple (only contains five highly optimized operators: 3x3 convolution, LeakyReLU, element-wise addition, element-wise multiplication and sigmoid). Since there are no other tricks, you can directly adopt EDSR framework to train the model.
