# Generative Recursive Mosaics

This project builds large-scale image mosaics using **Stable Diffusion XL** and classical computer vision techniques.

From afar, the result looks like a single coherent image. Up close, the image is made of many small, semantically meaningful tiles (e.g. flowers, fish, or stylized characters), each generated at inference time rather than taken from a fixed dataset.

---

## How it works

The pipeline is split into four main steps:

1. **Macro image generation**  
   A high-resolution base image is generated using Stable Diffusion XL.

2. **Color quantization**  
   The image is downsampled and clustered with K-Means in RGB space to obtain a limited color palette and a spatial map.

3. **Micro-tile generation**  
   For each color cluster, a semantic tile is generated with SDXL using color-conditioned prompts.  
   Tiles are filtered using simple statistics to avoid flat or overly noisy outputs.

4. **Mosaic assembly**  
   Tiles are placed according to the quantized map, and a lightweight color transfer aligns each tile with the local tone of the original image.

---

## Design choices

- Tiles are generated on the fly instead of being selected from a database  
- Subjects are chosen to support wide color variation without looking unnatural  
- Statistical filtering helps balance local detail and global readability  

The goal is to keep both the global image recognizable and the individual tiles visually meaningful.

---

## Tech stack

- Python  
- PyTorch  
- Stable Diffusion XL (Diffusers)  
- Scikit-learn  
- NumPy  


---

This project was developed for academic and experimental purposes, focusing on combining generative models with classical vision pipelines.


## Project structure

