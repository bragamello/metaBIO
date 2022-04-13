import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('brazil.jpg', 0)

# image size
lx = img.shape[0]
ly = img.shape[1]

# marked pixels
x, y = np.where(img != 255)
pixels = np.column_stack( (x,y) )

# loop over scales
nscales = 8
scales = 2**np.arange(nscales)[0:nscales]
Ns = []
for scale in scales:
    # counting box
    H, edges = np.histogramdd(pixels, bins=(np.arange(0,lx,scale),np.arange(0,ly,scale)))
    Ns.append(np.sum(H > 0))

# Linear fit
coeffs,cov = np.polyfit(np.log2(scales), np.log2(Ns), 1, cov=True)

# access fit and error estimation
#print('fractal dimension = ', -coeffs[0])
#print(np.sqrt(np.diag(cov)))

plt.plot(np.log2(scales),np.log2(Ns), 'o', color='blue', label = 'Box Counting')
plt.plot(np.log2(scales), np.polyval(coeffs,np.log2(scales)),'--',color='black', label = 'Fractal Dimension = '+str(round(-coeffs[0],3)))
plt.xlabel(r'$\log_2 \epsilon$')
plt.ylabel(r'$\log_2 N$')
plt.legend()
plt.show()
