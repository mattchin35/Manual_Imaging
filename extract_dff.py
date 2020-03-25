from __future__ import print_function
import numpy as np
import scipy as sp
import glob
import matplotlib.pyplot as plt
import os
import argparse
import sima
import sima.motion
import sima.segment
from sima.ROI import ROIList
import time


def movmean(x, k):
	"""A running mean for a 1D array. Edges are handled the same as matlab's movmean."""
	assert k < len(x)
	pad_width = k // 2
	x_pad = np.pad(x, pad_width, mode='constant')
	xc = np.convolve(x, np.ones(k), mode='full')
	xc = xc[pad_width:-pad_width]

	F = np.ones(len(xc)) * k
	F[:pad_width] -= np.flip(np.arange(pad_width)+1, axis=0)
	F[-pad_width:] -= np.arange(pad_width)+1
	return xc / F


def extract_dff(directory, plot=False, load_signal=False):
	hd, tl = os.path.split(directory)
	if load_signal:
		dff = np.load(os.path.join(hd, 'DFF'))		
	else:
		dataset = sima.ImagingDataset.load(directory)
		rois = ROIList.load(os.path.join(hd, 'manual', 'RoiSet.zip'), fmt='ImageJ')
		dataset.add_ROIs(rois, 'from_ImageJ')  # this immediately saves the ROIs
		signals = dataset.extract(rois, signal_channel='0', label='0') 
		signals = dataset.signals(channel='0')['0']

		# signals are in signals['raw']
		# first list index is for each cycle
		# second index is for each roi within a cycle (third is signal at time t)
		## convert each signal into a df/f trace
		k = 30
		baseline_ix = np.array([[0, 90], [240, 320]])
		traces = np.stack(signals['raw'], axis=1)
		dff = np.zeros_like(traces)
		nroi, ncycles, t = traces.shape

		# normalize the signal
		for i, roi in enumerate(traces):
			for j, cycle in enumerate(roi):
				cur_mins = []
				for b in baseline_ix:
					ITI = cycle[b[0]:b[1]]
					sroi = movmean(ITI, 30)
					cur_mins.append(np.min(sroi))
				bsl = np.min(cur_mins)
				roi_df = (cycle - bsl) / bsl
				dff[i,j,:] = roi_df  # rois x cycles x time

				# smooth the dff
				# sroi_df = sp.signal.savgol_filter(roi_df, 29, 3)
				# sroi_df = sp.ndimage.filters.gaussian_filter1d(roi_df,3)
				# dff[i,j,:] = sroi_df  # rois x cycles x time

		np.save(os.path.join(hd, 'manual', 'DFF'), dff)

	# check by plotting
	if plot:
		plt.subplots()
		plt.plot(traces[0,0,:], label='raw signal')
		plt.plot(dff[0,0,:], label='dF/F')
		plt.legend()
		plt.show()


if __name__ == '__main__':
	argParser = argparse.ArgumentParser()
	argParser.add_argument(
		"directory", action="store", type=str, default='',
		help="Directory with sima motion correction data.")
	argParser.add_argument(
		"-p", "--plot", action="store_true",
		help="Plot a sample dff signal.")
	argParser.add_argument(
		"-l", "--load_signal", action="store_true",
		help="Load existing roi signals.")
	args = argParser.parse_args()

	extract_dff(directory=args.directory, plot=args.plot, load_signal=args.load_signal)
