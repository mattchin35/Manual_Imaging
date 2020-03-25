from __future__ import print_function
import numpy as np
import argparse
import glob
import matplotlib.pyplot as plt
import os
import sima
import sima.motion
import sima.segment
from sima.ROI import ROIList
import time
import shutil


def motion_correction(path=os.curdir, overwrite=False, hidden_markov_mc=False):
	hd, tl = os.path.split(path)

	# print(os.listdir(path))
	a = glob.glob(os.path.join(path,'*.h5'))
	a = sorted(a)
	# print(a)

	#n_cycles = 2  # for testing
	n_cycles = len(a)
	sequences = [sima.Sequence.create('HDF5', i, 'tzyxc') for i in a[:n_cycles]]

	if hidden_markov_mc:  ## Hidden Markov MC
		mc_approach = sima.motion.HiddenMarkov2D(
		        granularity='row', num_states_retained=300, max_displacement=[100, 100])
		export_addon = '_HMM'
	else:  ## Plane tranlation MC
		mc_approach = sima.motion.PlaneTranslation2D(
		        max_displacement=[20, 20])
		export_addon = '_PT'


	dataset_name = os.path.join(path, tl+export_addon+'_mc.sima')
	avg_img_name = os.path.join(path, 'avgs_'+tl+export_addon+'.tif')
	if overwrite:
		try:
			shutil.rmtree(dataset_name)
			os.remove(avg_img_name)
		except OSError:
			pass

	st = time.clock()
	dataset = mc_approach.correct(sequences, dataset_name, trim_criterion=0.9)
	elapsed = time.clock() - st
	print(elapsed / n_cycles, 'seconds per cycle')

	dataset.export_averages([avg_img_name], fmt='TIFF16')


if __name__ == '__main__':
	argParser = argparse.ArgumentParser()
	argParser.add_argument(
        "directory", action="store", type=str, default='',
        help="Directory with .h5 files to motion correct.")
	argParser.add_argument(
        "-o", "--overwrite", action="store_true",
        help="Overwrite files if they already exist.")
	argParser.add_argument(
        "-hm", "--hidden_markov", action="store_true",
        help="Use hidden markov model based motion correction.")
	args = argParser.parse_args()

	motion_correction(path=args.directory, overwrite=args.overwrite, hidden_markov_mc=args.hidden_markov)

