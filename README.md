# Image Compressor

This is the implementation of the paper : <br/>

**Lossy Image Compression Using Singular Value Decomposition And Fast Fourier Transform** <br/>
Ayan Bag <br/>

Paper Link: [pdf](https://www.researchgate.net/profile/Ayan-Bag/publication/347983766_Lossy_Image_Compression_Using_Singular_Value_Decomposition_And_Fast_Fourier_Transform/links/5ff14b5892851c13fee303d3/Lossy-Image-Compression-Using-Singular-Value-Decomposition-And-Fast-Fourier-Transform.pdf)


## Overview
The image data is decomposed into three color channels red, blue and green. Each channels represent a matrix withthe values ranging from 0 to 255. Now, we compute an approximation of each color channel matrix, which take onlyfraction of space to store the data. Now selecting only r larger singular values of both orthogonal matrix and diagonalmatrix for each color channel to realize the first-stage compression of the image. Now, applying Fast Fourier Transformon each resultant matrix, to compress the data. Now the image is reconstructed using Inverse Fast Fourier Transform,and keeping only k% of data. The image is reconstructed. 

## Setup

Clone the repository and install other requirements:

```
git clone https://github.com/ayanbag/ImageCompressor.git
cd ImageCompressor
pip install -r requirements.txt
```

## Usage
For compressing the image:
```
python imgc.py <image_file_to_be_compressed>
```

## Cite as
>Bag, A., 2021. LOSSY IMAGE COMPRESSION USING SINGULAR VALUE DECOMPOSITION AND FAST FOURIER TRANSFORM. ([pdf](https://www.researchgate.net/profile/Ayan-Bag/publication/347983766_Lossy_Image_Compression_Using_Singular_Value_Decomposition_And_Fast_Fourier_Transform/links/5ff14b5892851c13fee303d3/Lossy-Image-Compression-Using-Singular-Value-Decomposition-And-Fast-Fourier-Transform.pdf))



**Bibtex :**
```
@article{bag2021lossy,
  title={LOSSY IMAGE COMPRESSION USING SINGULAR VALUE DECOMPOSITION AND FAST FOURIER TRANSFORM},
  author={Bag, Ayan},
  year={2021}
}
``` 
