# Learned-Image-Compression-with-GMM-and-Attention

This repository contains the code for reproducing the results and trained models, in the following paper:

Learned Image Compression with Discretized Gaussian Mixture Likelihoods and Attention Modules. [arXiv](https://arxiv.org/abs/2001.01568)

Zhengxue Cheng, Heming Sun, Masaru Takeuchi, Jiro Katto

## Paper Summary

Recently, learned compression methods exhibit a fast development trend with promising results. However, there is still a performance gap between learned compression algorithms and reigning compression standards, especially in terms of widely used PSNR metric. In this paper, we explore the remaining redundancy of recent learned compression algorithms. We have found accurate entropy models for rate estimation largely affect the optimization of network parameters and thus affect the rate-distortion performance. We propose to use discretized Gaussian Mixture Likelihoods to parameterize the distributions of latent codes, which can achieve a more accurate and flexible entropy model. Besides, we take advantage of recent attention modules and incorporate them into the network architecture to enhance the performance. Experimental results demonstrate our proposed method achieves a state-of-the-art performance compared to existing learned compression methods on both Kodak and high-resolution datasets. 

To our knowledge our approach is the first work to achieve comparable performance with latest compression standard Versatile Video Coding (VVC) regarding PSNR. More importantly, our approach can generate more visually pleasant results when optimized by MS-SSIM.

## Implementations

### Environment 
Python==3.6.4
Tensorflow==1.9.0

### Training
* usage:

### Test
* usage: 

## Reconstructed Samples

Comparisons of reconstructed samples are given in the following.

![](https://github.com/ZhengxueCheng/Learned-Image-Compression-with-GMM-and-Attention/blob/master/figures/visualizationKodim21Ver2.png)


## Evaluation Results

![](https://github.com/ZhengxueCheng/Learned-Image-Compression-with-GMM-and-Attention/blob/master/figures/RD.PNG)

## Notes




