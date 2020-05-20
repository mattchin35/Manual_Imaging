from __future__ import print_function
import numpy as np
import argparse
import glob
import matplotlib.pyplot as plt
import os
#import sima
#import sima.motion
#import sima.segment
#from sima.ROI import ROIList
import time
import shutil


def motion_correction(path=os.curdir, overwrite=False, hidden_markov_mc=False, output_path='', output_name=''):
	if len(path) > 1:
		assert output_path, "output_path required"
		assert output_name, "output_name required"

	if not output_name and len(path) == 1:
		_, output_name = os.path.split(path[0])
	if not output_path and len(path) == 1:
		output_path = path[0]
	
	a = [glob.glob(os.path.join(p,'*.h5')) for p in path]
	a = [sorted(p) for p in a]
	a = np.ravel(a).tolist()

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

	if not os.path.exists(output_path):
			os.mkdir(output_path)
	dataset_name = os.path.join(output_path, output_name+export_addon+'_mc.sima')
	
	if not os.path.exists(os.path.join(output_path, 'manual')):
			os.mkdir(os.path.join(path, 'manual'))
	avg_img_name = os.path.join(output_path, 'manual', 'avgs_'+output_name+export_addon+'.tif')
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
        "-ow", "--overwrite", action="store_true",
        help="Overwrite files if they already exist.")
	argParser.add_argument(
        "-hm", "--hidden_markov", action="store_true",
        help="Use hidden markov model based motion correction.")
	argParser.add_argument(
        "-op", "--output_path", action="store", default='',
        help="Path for output sima folder with motion corrected data.")
	argParser.add_argument(
        "-on", "--output_name", action="store", default='',
        help="Name for output sima folder with motion corrected data.")
	argParser.add_argument(
        "directories", nargs='*', action="store", type=str, default='',
        help="Directories with .h5 files to motion correct.")
	args = argParser.parse_args()
	print(args.directories)

	motion_correction(overwrite=args.overwrite, hidden_markov_mc=args.hidden_markov, 
		output_path=args.output_path, output_name=args.output_name, path=args.directories)

