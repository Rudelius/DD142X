from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys

import tensorflow as tf

from datasets import dataset_utils

_FRAC_VALIDATION = 0.2
_FRAC_TEST = 0.1
_RANDOM_SEED = 0
_NUM_SHARDS = 480

class ImageReader(object):
	"""Helper class that provides TensorFlow image coding utilities."""

	def __init__(self):
		# Initializes function that decodes RGB JPEG data.
		self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
		self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

	def read_image_dims(self, sess, image_data):
		image = self.decode_jpeg(sess, image_data)
		return image.shape[0], image.shape[1]

	def decode_jpeg(self, sess, image_data):
		image = sess.run(self._decode_jpeg,feed_dict={self._decode_jpeg_data: image_data})
		assert len(image.shape) == 3
		assert image.shape[2] == 3
		return image

def _get_filenames_and_classes(dataset_dir):
	directories = []
	class_names = []

	for filename in os.listdir(dataset_dir):
		path = os.path.join(dataset_dir, filename)
		if os.path.isdir(path) and filename[0]!='.':
			directories.append(path)
			class_names.append(filename)

	photo_categories = []
	for i, directory in enumerate(directories):
		photo_categories.append([])
		photo_filenames = []

		for filename in os.listdir(directory):
			if filename[-1]=='g' and filename[0]!='.':
				path = os.path.join(directory, filename)
				photo_filenames.append(path)

		photo_categories[i] = photo_filenames

	return photo_categories, sorted(class_names)

def _get_dataset_filename(dataset_dir, split_name, shard_id):
	output_filename = 'cancer_%s_%05d-of-%05d.tfrecord' % (split_name, shard_id, _NUM_SHARDS)
	return os.path.join(dataset_dir, output_filename)

def _convert_dataset(split_name, filenames, class_names_to_ids, dataset_dir):
	assert split_name in ['train', 'validation', 'test']

	num_per_shard = int(math.ceil(len(filenames) / float(_NUM_SHARDS)))
	with tf.Graph().as_default():
		image_reader = ImageReader()

		with tf.Session('') as sess:

			for shard_id in range(_NUM_SHARDS):
				output_filename = _get_dataset_filename(dataset_dir, split_name, shard_id)

				with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
					start_ndx = shard_id * num_per_shard
					end_ndx = min((shard_id+1) * num_per_shard, len(filenames))
					for i in range(start_ndx, end_ndx):
						sys.stdout.write('\r>> Converting images %d/%d shard %d file %s' % (i+1, len(filenames), shard_id, filenames[i]))
						sys.stdout.flush()

						try:
							# Read the filename:
							image_data = tf.gfile.GFile(filenames[i], 'rb').read()
							height, width = image_reader.read_image_dims(sess, image_data)
						except:
							print("\n Error at file " + filenames[i])
							exit()

						class_name = os.path.basename(os.path.dirname(filenames[i]))
						class_id = class_names_to_ids[class_name]

						example = dataset_utils.image_to_tfexample(image_data, b'jpg', height, width, class_id)
						tfrecord_writer.write(example.SerializeToString())

	sys.stdout.write('\n')
	sys.stdout.flush()

def _dataset_exists(dataset_dir):
	for split_name in ['train', 'validation']:
		for shard_id in range(_NUM_SHARDS):
			output_filename = _get_dataset_filename(dataset_dir, split_name, shard_id)
			if not tf.gfile.Exists(output_filename):
				return False
	return True

def run(dataset_dir):
	if not tf.gfile.Exists(dataset_dir):
		tf.gfile.MakeDirs(dataset_dir)

	if _dataset_exists(dataset_dir):
		print('Dataset files already exist. Exiting without re-creating them.')
		return
	photo_categories, class_names = _get_filenames_and_classes(dataset_dir)
	class_names_to_ids = dict(zip(class_names, range(len(class_names))))

	# Remove 10% from each category as test data.
	test_filenames = []
	remaining_filenames = []

	random.seed(_RANDOM_SEED)
	for category in photo_categories:
		num_test = int(round(len(category) * _FRAC_TEST))
		random.shuffle(category)
		test_filenames.extend(category[:num_test])
		remaining_filenames.extend(category[num_test:])

	# Remove 20% of all data as validation data.
	training_filenames = []
	validation_filenames = []

	random.shuffle(remaining_filenames)
	num_validation = int(round(len(remaining_filenames)*_FRAC_VALIDATION))
	training_filenames = remaining_filenames[num_validation:]
	validation_filenames = remaining_filenames[:num_validation]

	# Print data sample sizes.
	print("Training sample size n=" , len(training_filenames))
	print("Validation sample size n=" , len(validation_filenames))
	print("Test sample size n=" , len(test_filenames))


	# Convert to slim data format
	_convert_dataset('train', training_filenames, class_names_to_ids, dataset_dir)
	_convert_dataset('validation', validation_filenames, class_names_to_ids, dataset_dir)
	_convert_dataset('test', test_filenames, class_names_to_ids, dataset_dir)

	labels_to_class_names = dict(zip(range(len(class_names)), class_names))
	dataset_utils.write_label_file(labels_to_class_names, dataset_dir)

	#_clean_up_temporary_files(dataset_dir)
	print('\nFinished converting the Cancer dataset!')

if __name__ == "__main__":
    main()
