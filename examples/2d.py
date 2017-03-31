#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os
import sys
import numpy as np
import matplotlib.pyplot as pl

#d = os.path.dirname
#sys.path.insert(0, d(d(os.path.abspath(__file__))))  #this well change dir doesn't work
import george
from george.kernels import ExpSquaredKernel

np.random.seed(12345)

kernel = ExpSquaredKernel([3, 0.5], ndim=2)
#gp = george.HODLRGP(kernel, tol=1e-10)
gp = george.GP(kernel, solver = george.HODLRSolver，tol=1e-10);  #reference the tutorials

x, y = np.linspace(-5, 5, 62), np.linspace(-5, 5, 60) # generate workplane([-5,5],[-5,5])
x, y = np.meshgrid(x, y, indexing="ij") #generate grid, expand x to 60(ys for every x point)*62(x), y is 62x to y*60 ->indexing x,y as i,j
shape = x.shape #shape get the dimension of x=(62,60)
samples = np.vstack((x.flatten(), y.flatten())).T #flatten 2d to vector,row by row,1*3720 stack as vetical 2*3720,Trans to 3720,2
gp.compute(samples, 1e-4*np.ones(len(samples)), sort=False) #input sample as x, yerr as 1e-4, len(3720,2)=3720

print(len(samples)) #length of sample
i = george.utils.nd_sort_samples(samples) #sorted the samples return i

img = gp.get_matrix(samples[i]) #use sorted index i to rebuild a matrix
pl.imshow(img, cmap="gray", interpolation="nearest")  #should be gray for sample
pl.gca().set_xticklabels([])
pl.gca().set_yticklabels([])
pl.colorbar()
pl.savefig("2d-cov.png")

pl.clf()
z = np.empty(len(samples))
z[i] = gp.sample(samples[i])  #samples[i] means sorted sample
pl.pcolor(x, y, z.reshape(shape), cmap="gray")
pl.colorbar()
pl.savefig("2d.png")

import time

s = time.time()
gp.compute(samples, 1e-4*np.ones_like(z), sort=False) #pre-computer cov matrix,not sorted
print(time.time() - s)
s = time.time()
print(gp.lnlikelihood(z))
print(time.time() - s)

s = time.time()
gp.compute(samples, 1e-4*np.ones_like(z)) #sorted and compute cov matrix and lnlikelihood more stable
print(gp.lnlikelihood(z))
print(time.time() - s)

gp.kernel = ExpSquaredKernel([3.1, 0.6], ndim=2)  #from [3, .5]to[3, .6]

s = time.time()
gp.compute(samples, 1e-4*np.ones_like(z)) #new kernal
print(gp.lnlikelihood(z))
print(time.time() - s)

s = time.time()
gp.compute(samples[i], 1e-4*np.ones_like(z), sort=False)  #time not sorted
print(gp.lnlikelihood(z[i]))
print(time.time() - s)
