PheScape: dissecting multi-scale phenotypic landscapes of spatial niches via diffusion bridge and hierarchical alignment of unmatched samples
======
## Overview
PheScape is a diffusion-based computational framework designed to integrate multi-source transcriptomics (spatial transcriptomics, single-cell RNA-seq, and bulk RNA-seq) into unified phenotypic landscapes of spatial niches.

The resulting latent representations and soft correspondence matrix enable a range of downstream analyses, including spatial domain identification, cell-type deconvolution, clinical phenotype mapping, and multi-scale landscape construction. PheScape has been validated on 16 ST slices across four diseases—Alzheimer's disease, hepatocellular carcinoma, oral squamous cell carcinoma, and ovarian cancer—demonstrating its utility in identifying disease-associated niches, predicting survival outcomes, clinical stages, and drug sensitivity, and revealing subclone-specific prognosis. The framework is implemented in PyTorch and is publicly available with end-to-end reproducible tutorials.
## Requirements
Hardware resources used in this project.<br>
+ GPU NVIDIA GeForce RTX 3060<br>
+ CUDA Version 11.7<br><br>

You'll need to install the following packages in order to run the codes.<br>
+ python==3.8<br>
+ torch==2.0.0<br>
+ numpy==1.21.6<br>
+ pandas==2.0.0<br>
+ scanpy==1.9.3<br>
+ anndata==0.8.0<br>
+ scipy==1.10.1<br>
+ scikit-learn==1.2.2<br>
+ tqdm==4.65.0<br>
+ matplotlib==3.7.1<br>
+ R==4.2.2<br>
+ rpy2==3.5.10<br>

## Tutorial
For the step-by-step tutorial, please refer to: 
<br>
https://github.com/HYF01/PheScape/blob/main/PheScape/tutorial.md
<br>
A Jupyter Notebook of the tutorial is accessible from : 
<br>
https://github.com/HYF01/PheScape/blob/main/PheScape/tutorial.ipynb
<br>

