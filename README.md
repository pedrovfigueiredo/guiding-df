# Neural Path Guiding with Distribution Factorization

### [Paper](https://arxiv.org/pdf/2506.00839) | [Project Page](https://pedrovfigueiredo.github.io/projects/pathguiding/EGSR_2025_Importance_Sampling/index.html)

This is the official implementation of the paper, titled "Neural Path Guiding with Distribution Factorization", presented at EGSR 2025.

<img src="media/equaltimewspp-veachdoor.png" width="800px"/> <br/>
<!-- On the top, we show two relit images produced by NeRFFaceLighting (Jiang et al.), using the lighting extracted from images of individuals with fair and dark skin tones (shown on the right). 
As seen, NeRFFaceLighting produces relit images with inconsistent skin tones. Additionally, when distilling the EG3D triplane, NeRFFaceLighting tends to produce albedo maps that are biased towards lighter skin colors. 
Our method mitigates this bias and improves the consistency of the skin tone in relit images.
Note that even though we use the same latent vector to generate the results with EG3D, NeRFFaceLighting, and ours, there are variation in the images as the backbone EG3D network is finetuned separately in NeRFFaceLighting and ours. -->
Distribution Factorization (DF) factorizes the 2D directional distribution as a product of two 1D PDFs for efficient path guiding.
We show an equal-time comparison of our method against existing approaches for the Veach Door scene. Soecifically, we compare against PT (unidirectional path tracing), PPG [Müller et al. 2017], Variance [Rath et al. 2020], NIS [Müller et al. 2019], and NPM [Dong et al. 2023]. As seen, the two interpolation variants (nearest neighbor and linear interpolation) of our method produce images with significantly lower noise.

## Abstract
In this paper, we present a neural path guiding method to aid with Monte Carlo (MC) integration in rendering. Existing neural methods utilize distribution representations that are either fast or expressive, but not both. We propose a simple, but effective, representation that is sufficiently expressive and reasonably fast. Specifically, we break down the 2D distribution over the directional domain into two 1D probability distribution functions (PDF). We propose to model each 1D PDF using a neural network that estimates the distribution at a set of discrete coordinates. The PDF at an arbitrary location can then be evaluated and sampled through interpolation. To train the network, we maximize the similarity of the learned and target distributions. To reduce the variance of the gradient during optimizations and estimate the normalization factor, we propose to cache the incoming radiance using an additional network. Through extensive experiments, we demonstrate that our approach is better than the existing methods, particularly in challenging scenes with complex light transport.


## News
- **2025.06.26**: Repo is released.

## TODO List
- [ ] Code Release

## Citation
If our work is useful for your research, please consider citing:
```
@inproceedings{figueiredo25guidingdf,
  booktitle = {Eurographics Symposium on Rendering},
  editor = {Wang, Beibei and Wilkie, Alexander},
  title = {{Neural Path Guiding with Distribution Factorization}},
  author = {Figueiredo, Pedro and He, Qihao and Kalantari, Nima Khademi},
  year = {2025},
  publisher = {The Eurographics Association},
  ISSN = {1727-3463},
  ISBN = {978-3-03868-292-9},
  DOI = {10.2312/sr.20251178}
}
```
