
""" Prepare raw data. for eff. stereo matching

"""



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from builtins import input

#import system things

import tensorflow as tf
import numpy as np
import pandas as pd
import os

from scipy import misc


train_path = './training'
util_path =  './preprocess/debug_15/'
NUM_CHANNELS = 1
dataset = 'kitti2012'

def gen_raw_data(util_path, half_range, num_tr_samples, num_val_samples):


	dt1=np.dtype([('file_id', 'f4'), ('direction', 'f4'), ('center_x', 'f4'), ('center_y' , 'f4'), ('right_center_x', 'f4')])
	raw_tr_loc=np.fromfile(util_path + 'tr_160_18_100.bin',dtype=dt1)
	df_tr_loc=pd.DataFrame(raw_tr_loc.tolist(), columns=raw_tr_loc.dtype.names)

	dt2=np.dtype([('file_id', 'f4'), ('direction', 'f4'), ('center_x', 'f4'), ('center_y' , 'f4'), ('right_center_x', 'f4')])
	raw_val_loc=np.fromfile(util_path + 'val_34_18_100.bin',dtype=dt2)
	df_val_loc=pd.DataFrame(raw_val_loc.tolist(), columns=raw_val_loc.dtype.names)

	tr_loc_perm = np.random.RandomState(seed=242).permutation(df_tr_loc.shape[0])

	val_loc_perm = np.random.RandomState(seed=242).permutation(df_val_loc.shape[0])


	all_labels_train = np.ones([df_tr_loc.shape[0],1])*(half_range)

	all_labels_val = np.ones([df_val_loc.shape[0],1])*(half_range)

	#tr_loc_perm = tf.convert_to_tensor(tr_loc_perm, dtype=tf.float32)

	#val_loc_perm = tf.convert_to_tensor(val_loc_perm, dtype=tf.float32)

	# convert pds to numpy arrays & permutate
	df_tr_np = df_tr_loc.values

	df_tr_np.astype(dtype=int,copy=False)

	df_val_np = df_val_loc.values

	df_val_np.astype(dtype=int,copy=False)

	for i in range(2, 5):
		df_tr_np[:, i] -= 1
		df_val_np[:, i] -= 1


	df_np_perm = df_tr_np[tr_loc_perm[0:num_tr_samples]]

	all_labels_train = all_labels_train[0:num_tr_samples]

	df_np_val_loc = df_val_np[val_loc_perm[0:num_val_samples]]

	all_labels_val = all_labels_val[0:num_val_samples]

	return df_np_perm, all_labels_train, df_np_val_loc, all_labels_val


def gen_filenames(util_path, train_path, num_tr_imgs, num_val_imgs):

	dt=np.dtype([('file_id', 'f4')])
	raw_fid=np.fromfile(util_path + 'myPerm.bin',dtype=dt)
	df_fid=pd.DataFrame(raw_fid.tolist(), columns=raw_fid.dtype.names)

	left_images_filenames = []
	right_images_filenames = []
	# Reading left and right images filenames in two separate lists
	for i in range(num_tr_imgs+num_val_imgs):
		left_images_filenames.append('{}/image_2/{:06d}_10.png'.format(train_path, df_fid['file_id'].apply(int)[i]))
		right_images_filenames.append('{}/image_3/{:06d}_10.png'.format(train_path, df_fid['file_id'].apply(int)[i]))
		print(left_images_filenames)

	# Converting lists to tensors
	left_images_filenames = tf.convert_to_tensor(left_images_filenames)
	right_images_filenames = tf.convert_to_tensor(right_images_filenames)

	return left_images_filenames, right_images_filenames

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def load_images(util_path, data_root, num_tr_img, num_val_img):

	dt=np.dtype([('file_id', 'f4')])
	raw_fid=np.fromfile(util_path + 'myPerm.bin',dtype=dt)
	df_fid=pd.DataFrame(raw_fid.tolist(), columns=raw_fid.dtype.names)

	print('Number of training images {}'.format(num_tr_img))
	print('Number of Validation images {}'.format(num_val_img))

	ldata = {}
	rdata = {}

	for i in range(num_tr_img+num_val_img):

		fn = df_fid['file_id'].apply(int)[i]

		#print(i)
		if dataset == 'kitti2015':
			l_img = rgb2gray(misc.imread(('%s/image_2/%06d_10.png') % (data_root, fn)))
			r_img = rgb2gray(misc.imread(('%s/image_3/%06d_10.png') % (data_root, fn)))
		elif dataset == 'kitti2012':
			l_img = misc.imread(('%s/image_0/%06d_10.png') % (data_root, fn))
			r_img = misc.imread(('%s/image_1/%06d_10.png') % (data_root, fn))



		l_img = (l_img - l_img.mean()) / l_img.std()
		r_img = (r_img - r_img.mean()) / r_img.std()

		ldata[fn] = l_img.reshape(l_img.shape[0], l_img.shape[1], NUM_CHANNELS).astype(np.float32)
		rdata[fn] = r_img.reshape(r_img.shape[0], r_img.shape[1], NUM_CHANNELS).astype(np.float32)

	return ldata, rdata
