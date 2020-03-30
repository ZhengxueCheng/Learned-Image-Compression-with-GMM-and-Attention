# Learned-Image-Compression-with-GMM-and-Attention

This repository contains the code for reproducing the results with trained models, in the following paper:

Learned Image Compression with Discretized Gaussian Mixture Likelihoods and Attention Modules. [arXiv](https://arxiv.org/abs/2001.01568), CVPR2020.

Zhengxue Cheng, Heming Sun, Masaru Takeuchi, Jiro Katto

## Paper Summary

Recently, learned compression methods exhibit a fast development trend with promising results. However, there is still a performance gap between learned compression algorithms and reigning compression standards, especially in terms of widely used PSNR metric. In this paper, we explore the remaining redundancy of recent learned compression algorithms. We have found accurate entropy models for rate estimation largely affect the optimization of network parameters and thus affect the rate-distortion performance. We propose to use discretized Gaussian Mixture Likelihoods to parameterize the distributions of latent codes, which can achieve a more accurate and flexible entropy model. Besides, we take advantage of recent attention modules and incorporate them into the network architecture to enhance the performance. Experimental results demonstrate our proposed method achieves a state-of-the-art performance compared to existing learned compression methods on both Kodak and high-resolution datasets. 

To our knowledge our approach is the first work to achieve comparable performance with latest compression standard Versatile Video Coding (VVC) regarding PSNR. More importantly, our approach can generate more visually pleasant results when optimized by MS-SSIM.

### Environment 

* Python==3.6.4

* Tensorflow==1.14.0

* [RangeCoder](https://github.com/lucastheis/rangecoder)

```   
    pip3 install range-coder
```

* [Tensorflow-Compression](https://github.com/tensorflow/compression) ==1.2

```
    pip3 install tensorflow-compression or 
    pip3 install tensorflow_compression-1.2-cp36-cp36m-manylinux1_x86_64.whl
```

### Test Usage

* Download the pre-trained [models](https://drive.google.com/open?id=19b92ey1g30R2OvWupekLQNb3TjHs5HLX) (this model is optimized by MS-SSIM using lambda = 14) and unzip it.

* Put your images to the directory valid/ and run the py files


```
    python3 encoder.py
```
```
    python3 decoder.py
```


## Reconstructed Samples

Comparisons of reconstructed samples are given in the following.

![](https://github.com/ZhengxueCheng/Learned-Image-Compression-with-GMM-and-Attention/blob/master/figures/visualizationKodim21Ver2.png)


## Evaluation Results

![](https://github.com/ZhengxueCheng/Learned-Image-Compression-with-GMM-and-Attention/blob/master/figures/RD.PNG)

## Notes

This implementations are not original codes of our CVPR2020 paper, because original code is based on Tensorflow 1.9.0 and many features have been removed. This repo is a re-implementation, but the core codes are almost the same and results are also consistent with original results. This repo is also submitted to CVPR Workshop and Challenge on Leanred Image Challenge ([CLIC] (http://www.compression.cc/)) with the entry Kattolab in the Leaderboard.

If you think it is useful for your reseach, please cite our CVPR2020 paper. Our original RD data in the paper is contained in the folder RDdata/.




