from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from math import log10, sqrt
from matplotlib.image import imread
import matplotlib.pyplot as plt
import os
from skimage import metrics
import cv2
import sys
import itertools
import threading
import time
import os

filename = sys.argv[-1]
done = False
#here is the animation
def animate():
    for c in itertools.cycle(['|', '/', '-', '\\']):
        if done:
            break
        sys.stdout.write('\rCompressing ' + c)
        sys.stdout.flush()
        time.sleep(0.1)

t = threading.Thread(target=animate)
t.start()

def PSNR(original, compressed): 
    mse = np.mean((original - compressed) ** 2) 
    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                  # Therefore PSNR have no importance. 
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse)) 
    return psnr

#plt.rcParams['figure.figsize'] = [10, 10]
#plt.rcParams.update({'font.size': 10})

image = np.array(Image.open(filename))

image = image / 255
row, col, _ = image.shape

image_red = image[:, :, 0]
image_green = image[:, :, 1]
image_blue = image[:, :, 2]

U_r, d_r, V_r = np.linalg.svd(image_red, full_matrices=True)
U_g, d_g, V_g = np.linalg.svd(image_green, full_matrices=True)
U_b, d_b, V_b = np.linalg.svd(image_blue, full_matrices=True)

#Parameters
k=190
keep=0.5

U_r_k = U_r[:, 0:k]
V_r_k = V_r[0:k, :]
U_g_k = U_g[:, 0:k]

V_g_k = V_g[0:k, :]
U_b_k = U_b[:, 0:k]
V_b_k = V_b[0:k, :]

d_r_k = d_r[0:k]
d_g_k = d_g[0:k]
d_b_k = d_b[0:k]

image_red_a = np.dot(U_r_k, np.dot(np.diag(d_r_k), V_r_k))
image_green_a = np.dot(U_g_k, np.dot(np.diag(d_g_k), V_g_k))
image_blue_a = np.dot(U_b_k, np.dot(np.diag(d_b_k), V_b_k))

rt = np.fft.fft2(image_red_a)
rtsort = np.sort(np.abs(rt.reshape(-1))) # sort by magnitude

gt=np.fft.fft2(image_green_a)
gtsort=np.sort(np.abs(gt.reshape(-1)))

bt=np.fft.fft2(image_blue_a)
btsort=np.sort(np.abs(bt.reshape(-1)))

threshr = rtsort[int(np.floor((1-keep)*len(rtsort)))]
indr = np.abs(rt)>threshr         
rtlow = rt * indr               
rlow = np.fft.ifft2(rtlow).real
    
threshg = gtsort[int(np.floor((1-keep)*len(gtsort)))]
indg = np.abs(gt)>threshg         
gtlow = gt * indg               
glow = np.fft.ifft2(gtlow).real
    
threshb = btsort[int(np.floor((1-keep)*len(btsort)))]
indb = np.abs(bt)>threshb         
btlow = bt * indb               
blow = np.fft.ifft2(btlow).real
    
image_reconstructed = np.zeros((row, col, 3))
image_reconstructed[:, :, 0] = rlow
image_reconstructed[:, :, 1] = glow
image_reconstructed[:, :, 2] = blow
    
value = PSNR(image, image_reconstructed) 
s = metrics.structural_similarity(image, image_reconstructed,multichannel=True)
    
name=filename.split(".")
fig = plt.figure(figsize=(5,4))
imgplot = plt.imshow(image_reconstructed)
plt.axis('off')
plt.tight_layout()
plt.savefig(name[0]+"_compressed."+name[-1])

value = PSNR(image, image_reconstructed) 
s = metrics.structural_similarity(image, image_reconstructed,multichannel=True)

done = True


print("\n")
print("Image Compressed!!!")
print("PSNR value : ","%.2f" % value,"dB") 
print("SSIM ratio : ","%.2f" % s) 










