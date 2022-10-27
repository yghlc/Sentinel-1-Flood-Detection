#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : {Conor Simmons }
# Edited By   : {Ryan Cassotto}
# Created Date: 2022/10/24
# version ='1.0'
# ---------------------------------------------------------------------------
""" Bimodal Thresholding Module"""  
# ---------------------------------------------------------------------------

from image_proc_module_v01 import Image_proc
import numpy as np
import rasterio
from skimage import filters
from scipy.interpolate import CubicSpline
from scipy.optimize import fsolve
#import csv
import math
import matplotlib.pyplot as plt
import os

class BimodalThreshold(Image_proc):

    def __init__(self, c_bounds):
        self.c_bounds = c_bounds
#        self.deg_const = 0.000089831528412*4000 # a constant physical window size in units of degrees# Clay's original window size (based on 10 m GRD data)
        self.deg_const = 0.000089831528412*4000 # a constant physical window size in units of degrees; this works reasonably well for Pyeongchang. May need to 
        ## autonomosouly select image size or nFactor Size (e.g. 4000) based on size of input image. 4000 works well for entire Sentiel-1 image, 400 is better for 
        ## IARPA sites. 
         

    """ Helper functions """

    # img is an Image object
    # can change phys_size to allow the window to be a different physical size on the ground (units of degrees)
    def get_window_size(self, img, phys_size=None):
        if phys_size == None: phys_size = self.deg_const
        x, y = rasterio.transform.xy(img.raster.transform, 0, 0)
        _, intvl = rasterio.transform.rowcol(img.raster.transform, x+phys_size, y)
        while intvl % 8 != 0:       # note "%" is a python operator for modulus; changing this to 4 and 2 did not have an effect
            intvl += 1
        return intvl, int(intvl/8)

    def gaussian_convolution(self, y):
        gs = [0.2261, 0.5478, 0.2261] # from Uddin paper
        pad_y = np.zeros(len(y)+2)
        pad_y[1:-1] = y
        new_y = np.zeros(len(y))
        for k in range(1,len(pad_y)-1):
            t1 = pad_y[k-1] * gs[0] # term 1
            t2 = pad_y[k] * gs[1] # term 2
            t3 = pad_y[k+1] * gs[2] # term 3
            new_y[k-1] = t1+t2+t3
        return new_y

    def count_peaks(self, x, y, cs, dcs, eps_tol=1e-16):
        peak_cnt = 0
        valley_cnt = 0
        peaks = []
        valleys = []
        peak_heights = []
        for i in range(1,len(y)):
            if np.abs(y[i-1]) < eps_tol: continue
            else:
                d1 = dcs(x[i-1])
                d2 = dcs(x[i])
                sign1 = d1 / np.abs(d1)
                sign2 = d2 / np.abs(d2)
                test = sign1*sign2
                if test == -1.:
                    if sign1 == 1.:
                        peak_cnt += 1
                        peaks.append(fsolve(dcs, x[i]))
                        peak_heights.append(y[i])
                    else:
                        valley_cnt += 1
                        valleys.append(fsolve(dcs, x[i])[0])
        return peak_cnt, valley_cnt, peaks, valleys, peak_heights

    # legacy code
    def p1_p2_m1_m2_sb_sw_st(self, th, hist, M):
        t_vector = np.arange(0,256,1)
        # splicing
        t_lower = t_vector[0:th]
        t_higher = t_vector[th+1:-1]
        h_lower = hist[0:th]
        h_higher = hist[th+1:]
        # computations
        p1 = np.divide(np.sum(h_lower), M)
        p2 = np.divide(np.sum(h_higher), M)
        m1 = np.sum(np.divide(np.multiply(t_lower, h_lower), M*p1))
        m2 = np.sum(np.divide(np.multiply(t_higher, h_higher), M*p2))
        s12 = np.sum(np.multiply(np.square(np.subtract(t_lower, m1)), np.divide(np.divide(h_lower, M), p1)))
        s22 = np.sum(np.multiply(np.square(np.subtract(t_higher, m2)), np.divide(np.divide(h_higher, M), p2)))
        # scalar arithmetic
        sb2 = p1*p2*(m1-m2)**2
        sw2 = p1*s12 + p2*s22
        st2 = sb2 + sw2
        Bt = sb2/st2
        return Bt

    # legacy code
    def block_arrays_list(self, base_arr, block_dim):
        row, col = base_arr.shape
        num_row_blocks = math.ceil(row / block_dim)
        num_col_blocks = math.ceil(col / block_dim)
        row_indexes = np.arange(0, num_row_blocks * block_dim + 1, block_dim)
        col_indexes = np.arange(0, num_col_blocks * block_dim + 1, block_dim)
        indicies = []
        for i in range(0, len(row_indexes) - 1):
            row_s, row_e = row_indexes[i], row_indexes[i + 1]
            row_item = str(row_s) + ':' + str(row_e)
            inner = []
            for j in range(0, len(col_indexes) - 1):
                col_s, col_e = col_indexes[j], col_indexes[j + 1]
                col_item = str(col_s) + ':' + str(col_e)
                item_use = row_item + ',' + col_item
                inner.append(item_use)
            indicies.append(inner)
        indicies = np.array(indicies)
                
        return indicies, num_row_blocks, num_col_blocks

    # legacy code
    def block_arrays(self, base_arr, block_dim):
        squares_list, num_row_blocks, num_col_blocks = self.block_arrays_list(base_arr, block_dim)
        subset_arrays = []
        for j in range(0, num_row_blocks):
            for i in range(0, num_col_blocks):
                square_list_str = squares_list[j, i]
                square_list_row_s, square_list_row_e = int(square_list_str.split(',')[0].split(':')[0]), int(
                    square_list_str.split(',')[0].split(':')[1])
                square_list_col_s, square_list_col_e = int(square_list_str.split(',')[1].split(':')[0]), int(
                    square_list_str.split(',')[1].split(':')[1])
                subset_array = base_arr[square_list_row_s:square_list_row_e, square_list_col_s:square_list_col_e]

                subset_arrays.append(subset_array)
        return subset_arrays, num_row_blocks, num_col_blocks

    # legacy code
    def normalize_array_and_bin(self, subset_array_flat, N):
        subset_array_norm = (subset_array_flat - min(subset_array_flat))/(max(subset_array_flat) - min(subset_array_flat))
        bins = np.linspace(0, 1, N) # spacing of 256 discrete points between 0 and 1
        bin_count = np.histogram(subset_array_norm, bins)[0] # returns the count in each bin
        M = len(subset_array_norm)
        return bin_count, M # returns t=[0 1 2 ... 255], histogram, and number of elements in subarray

    """ Driver to process using LM and Otsu """

    def otsu_and_lm(self, images, out_dir, ptf=True, v=0.1, block_dim=4000, s=500, B_thresh=0.75, smoothing_tol=30, verbose=False):
        otsus = []
        lms = []
        eps = np.finfo(np.float).eps # constant for machine epsilon
  
        for img in images:
            otsu_part = []
            lm_part = []
            ncount = 0
            print('Working on %s' % {img.path})

            if ptf: band = np.where(img.band > 0., img.band**v, 0.) # power transform
            else: band = img.band
            

            subset_arrays, _, _ = self.block_arrays(band, block_dim)  # Sub-tiles or window size
            for subset_array, i in zip(subset_arrays, range(len(subset_arrays))):    # loop over subtiles
                tiles, _, _ = self.block_arrays(subset_array, s)  # break up sub-tiles into s x s sub-arrays
                for tile in tiles:

                    tile_flat = tile.flatten()
                    tile_flat = tile_flat[tile_flat != 0]

                    # handle case of only 0's
                    if len(tile_flat) == 0: continue

                    bin_count, M = self.normalize_array_and_bin(tile_flat, 256)
                    t_vector = np.arange(0,256,1) # vector in interval [0, 255]
                    B_vec = []
                    for th in t_vector:
                        B = self.p1_p2_m1_m2_sb_sw_st(th, bin_count, M)
                        B_vec.append(B)
                    B_vec = np.array(B_vec)
                    if len(B_vec) > 0: max_B = np.max(B_vec)
                    else: max_B = 0

                    # another condition to check if tile is in a no-data zone
                    min_cnt = np.sum(tile == np.min(tile))

                    # if the BCV condition is met
                    if max_B > B_thresh and min_cnt < 100:
                        # Otsu method
                        otsu_threshold = filters.threshold_otsu(image=tile_flat, nbins=256)  # run threshold based on Otsu's method
                        otsu_part.append(otsu_threshold)

                        # LM method
                        y, bins = np.histogram(tile_flat, bins=256)  # y = bin count; bins = bin edges
                        x = (bins[1:]+bins[:-1]) / 2    # x = bin center or median value of bin edges
                        if verbose:
                            fig, ax = plt.subplots(ncols=3, figsize=(20,6))
                            ax[0].imshow(tile, cmap='gray')
                            ax[0].set_title('Tile of amplitude image')
                            ax[1].plot(x,y)
#                            ax[1].set_title('Histogram with $B={max_B}$')
                            ax[1].set_title('Histogram with B = {:.2f}'.format(max_B))

                        cs = CubicSpline(x, y)
                        dcs = cs.derivative()
                        peak_cnt, valley_cnt, peaks, valleys, peak_heights = self.count_peaks(x, y, cs, dcs)
                        count = 0
                        while peak_cnt > 2 and count < smoothing_tol:
                            count += 1
                            y = self.gaussian_convolution(y)
                            cs = CubicSpline(x, y)
                            dcs = cs.derivative()
                            peak_cnt, valley_cnt, peaks, valleys, peak_heights = self.count_peaks(x, y, cs, dcs)
                        lm_threshold = None

                        # find the leftmost of the two highest peaks plus the one after that to split
                        if peak_cnt > 1: 
                            peakh1 = np.max(peak_heights)
                            pidx1 = np.argmax(peak_heights)
                            peak_heights_alt = peak_heights
                            peak_heights_alt[pidx1] = 0.
                            peakh2 = np.max(peak_heights_alt)
                            pidx2 = np.argmax(peak_heights_alt)
                            peak_list = [peaks[pidx1], peaks[pidx2]]
                            pidx_list = [pidx1, pidx2]
                            peak1 = np.min(peak_list)
                            meta_idx = np.argmin(peak_list)
                            pidx = pidx_list[meta_idx] + 1
                            peak2 = peaks[pidx]

                            for valley in valleys:
                                if valley > peak1 and valley < peak2: lm_threshold = valley
                        if lm_threshold != None:
                            lm_part.append(lm_threshold)

                        if verbose:
                            ax[2].plot(x, y)
#                            ax[2].set_title('Otsu (red): {otsu_threshold}; LM (green): {lm_threshold}')
                            print('OTSU threshold: ', otsu_threshold)
                            print('LM threshold: ', lm_threshold)
                            try:
                                
                                ax[2].set_title('Otsu (red): {:.2f}'.format(otsu_threshold) + '; LM (green): {:.2f}'.format(lm_threshold))
                                ax[2].scatter(otsu_threshold, cs(otsu_threshold), c='r')
                                ax[2].scatter(lm_threshold, cs(lm_threshold), c='g')
                            except:
                               ax[2].set_title('Otsu (red): {:.2f}'.format(otsu_threshold))
                               ax[2].scatter(otsu_threshold, cs(otsu_threshold), c='r')
                               
                               
                            #plt.show()
#                            print('Working on %s' % {img.path})    
                            ncount = ncount + 1
                            infilename = (img.path).rsplit('/',1)[1]   
                            print("infilename: ", infilename)
                            if "Sigma0_VV" in infilename:
                                out_pngfilename = infilename.replace('_Sigma0_VV.tif','_SAR_AMP_THRESH_sArray_' + str(ncount) + '.png') 
                            else:
                                out_pngfilename = infilename.replace('_Sigma0_VH.tif','_SAR_AMP_THRESH_sArray_' + str(ncount) + '.png')                                
#                            out_pngfilename = infilename.replace('_Sigma0_VV.tif','_SAR_AMP_THRESH_sArray_' + str(ncount) + '.png') 
                            outfilename_and_path = (out_dir + '/' + out_pngfilename)
                           
                            

                            supTitleStr= out_pngfilename
                            print(supTitleStr)
                            plt.suptitle(supTitleStr)
                            plt.savefig(outfilename_and_path)
            otsus.append(otsu_part)
            lms.append(lm_part)
        return otsus, lms