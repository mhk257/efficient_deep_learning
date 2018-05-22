
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import control_flow_ops
import time
from scipy import misc


from prep_raw import *
from net1_w37 import *

BATCH_SIZE = 128
TRAIN_ITERS_PER_EPOCH = 100
MAX_EPOCHS = 400
EPOCH_STEP = 40


MODEL_PATH = "model/"
MAX_ITERS = 40 #40000
learning_rate = 0.01
learning_rate_decay_factor = 5

train_path = './training'
util_path =  './preprocess/debug_15/'

psz = 18
num_tr_imgs = 160
num_val_imgs = 34
patchsz = 18
pSize = 2*patchsz + 1
half_range = 100

IMAGE_HEIGHT = pSize
IMAGE_WIDTH = pSize
NUM_CHANNELS = 1
num_tr_samples = 512
num_val_samples = 2560000

train_mode = False



def py_extract_patch(input_vars):
	#print(x)
	#print("Actual input --> {} {} {}".format(input_vars[2],input_vars[3],input_vars[4]))

	x, c_x, c_y, rc_x = input_vars[0].astype(np.int32), input_vars[2].astype(np.int64), input_vars[3].astype(np.int64), input_vars[4].astype(np.int64)

	rc_y = c_y

	#print("center_x-->{} center_y-->{} right_center_x-->{}".format(c_x, c_y, rc_x))


	left_image_patch = left_images[x][c_y - patchsz : c_y + patchsz + 1, c_x - patchsz : c_x + patchsz + 1, :]

	right_image_patch = right_images[x][rc_y - patchsz : rc_y + patchsz + 1, rc_x - patchsz -  half_range : rc_x + patchsz + half_range + 1, :]

	return (left_image_patch, right_image_patch)




def _extract_Left_right_patch(train_input_queue, input_label):


	train_left_patch, train_right_patch = tf.py_func(py_extract_patch, [train_input_queue], [tf.float32, tf.float32])


	train_left_patch.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])

	train_right_patch.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH + half_range*2, NUM_CHANNELS])


	train_label = input_label

	return train_left_patch, train_right_patch, train_label


left_images, right_images = load_images(util_path,train_path, num_tr_imgs, num_val_imgs)

batch_size = tf.placeholder(tf.int64)

is_training = tf.placeholder(tf.bool, name='is_training')


def create_dataset(data_split, util_path, half_range, batch_size):

	if data_split == 'train':
		patches_data, all_labels, _, _ = gen_raw_data(util_path, half_range, num_tr_samples, num_val_samples)
		patches_data = patches_data.tolist()
		all_labels = all_labels.tolist()
	elif data_split == 'val':
		_, _, patches_data, all_labels = gen_raw_data(util_path, half_range, num_tr_samples, num_val_samples)
		patches_data = patches_data.tolist()
		all_labels = all_labels.tolist()

	def prod():
		for row_input, label_input in zip(patches_data, all_labels):
			row_input = np.asarray(row_input, dtype=np.float32)
			label_input = np.asarray(label_input, dtype=np.float32)
			#print('Actual values {}'.format(row_input[0:]))
			#print(row_input.shape)
			yield row_input, label_input

	dataset = tf.data.Dataset.from_generator(prod, (tf.float32, tf.float32), (tf.TensorShape([5,]), tf.TensorShape([1,])))

	dataset = dataset.map(map_func=_extract_Left_right_patch, num_parallel_calls=2)

	dataset = dataset.batch(batch_size)
	#dataset = dataset.repeat(1)
	#dataset = dataset.prefetch(1)

	return dataset

# create datasets for training and evaluation
train_ds = create_dataset('train', util_path, half_range, BATCH_SIZE).repeat().prefetch(1)
val_ds = create_dataset('val', util_path, half_range, batch_size)



# handle and iterator
handle = tf.placeholder(tf.string, shape=[])

iterator = tf.data.Iterator.from_string_handle(handle, train_ds.output_types, train_ds.output_shapes)

train_left_batch, train_right_batch, train_label_batch = iterator.get_next()

training_iterator = train_ds.make_one_shot_iterator()
validation_iterator = val_ds.make_initializable_iterator()


sia_graph = tf.Graph()

with sia_graph.as_default():

	with tf.variable_scope("eff_siamese_model"):

		left_output = sia_net_slim(train_left_batch, is_training, reuse=False)

		right_output = sia_net_slim(train_right_batch, is_training, reuse=True)


		left_output = tf.squeeze(left_output, [1])

		right_output = tf.squeeze(right_output, [1])


		right_output = tf.transpose(right_output, perm=[0, 2, 1])


		left_right_dot = tf.matmul(left_output, right_output)

		logits = tf.contrib.layers.flatten(left_right_dot)

		#probs = tf.nn.log_softmax(logits)

		y_pred = tf.argmax(logits, axis=1)

		y_true = tf.cast(tf.squeeze(train_label_batch),tf.int64)

		curr_count = tf.reduce_sum(tf.cast(tf.less_equal(tf.abs(y_pred - y_true), 3), tf.float32)) # 3 pixel error

		curr_accuracy = curr_count / tf.cast(batch_size, tf.float32) * 100

if train_mode:

	with tf.variable_scope("loss"):

		loss = eff_stereo_loss2(logits, tf.squeeze(train_label_batch), BATCH_SIZE, half_range)
		#reg_losses = slim.losses.get_regularization_losses(scope='eff_siamese_model')
		#reg_loss = tf.reduce_sum(reg_losses)
		#loss = loss + reg_loss

## OPTIMIZER
	with tf.variable_scope("optimizer"):
		lr = tf.placeholder(tf.float32)
		global_step = tf.get_variable("global_step", [], initializer=tf.constant_initializer(0.0), trainable=False)
		optimizer = tf.train.AdagradOptimizer(lr)
		train_op = slim.learning.create_train_op(loss, optimizer, global_step=global_step)

		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		if update_ops:
			updates = tf.group(*update_ops)
			loss = control_flow_ops.with_dependencies([updates], loss)


	sess = tf.Session()

	training_handle = sess.run(training_iterator.string_handle(), feed_dict={batch_size: BATCH_SIZE})
	validation_handle = sess.run(validation_iterator.string_handle(), feed_dict={batch_size: BATCH_SIZE})

	print('Train iterator initialized....')
	sess.run(tf.global_variables_initializer(), feed_dict={is_training: True})
	print('Intialized global variables....')

	#setup tensorboard
	acc_loss = tf.placeholder(tf.float32, shape=())
	acc_accuracy = tf.placeholder(tf.float32, shape=())

	tf.summary.scalar('train/loss', acc_loss)
	tf.summary.scalar('train/laccuracy', acc_accuracy)
	for var in tf.trainable_variables():
		tf.summary.histogram(var.op.name, var)

	train_summary_op = tf.summary.merge_all()

	a = tf.summary.scalar('val/loss', acc_loss)
	b = tf.summary.scalar('val/accuracy', acc_accuracy)
	val_summary_op = tf.summary.merge([a, b])

	writer = tf.summary.FileWriter('train.log', graph=tf.get_default_graph())

	saver = tf.train.Saver()

	print('All Intitialization done...')

	start_session_time = time.time()

	best_val_accuracy = np.float32(0.0)

	for j in range(MAX_EPOCHS):

		tot_acc = np.float32(0.0)

		tot_loss = np.float32(0.0)

		for i in range(TRAIN_ITERS_PER_EPOCH):

			_, curr_loss, curr_acc, step = sess.run([train_op, loss, curr_count, global_step], feed_dict={batch_size: BATCH_SIZE, is_training: True, lr: learning_rate, handle: training_handle})

			#print('Train. Iter.>{} Curr. loss>{:8.5f} Curr Acc.>{}'.format(i,curr_loss,curr_acc))

			#print(curr_acc)
			#print(type(curr_acc))

			tot_acc += curr_acc
			tot_loss += curr_loss

		train_perf = (tot_acc / (TRAIN_ITERS_PER_EPOCH * BATCH_SIZE)) * 100

		train_loss = (tot_loss / (TRAIN_ITERS_PER_EPOCH))

		summary_str = sess.run(train_summary_op, feed_dict={acc_loss: train_loss, acc_accuracy: train_perf})
		writer.add_summary(summary_str, j)

		kk = 0
		sess.run(validation_iterator.initializer, feed_dict={batch_size: BATCH_SIZE})
		val_acc = np.float32(0.0)
		val_loss = np.float32(0.0)
		while True:
			try:
				curr_loss_val, curr_acc_val, step = sess.run([loss, curr_count, global_step], feed_dict={batch_size: BATCH_SIZE, is_training: False, handle: validation_handle})
				val_acc += curr_acc_val
				val_loss += curr_loss_val
				kk += 1
			except tf.errors.OutOfRangeError:
				print("Number of Val. samples-->{} Step-->{}".format(kk,step))
				val_loss = val_loss / kk
				val_accuracy = (val_acc / (kk * BATCH_SIZE)) * 100
				break

		summary_str = sess.run(val_summary_op, feed_dict={acc_loss: val_loss, acc_accuracy: val_accuracy})
		writer.add_summary(summary_str, j)

		print("Tr. EPOCH>{} Tr. Acc>{:5.5f} FullVal. Acc>{:5.5f} Train.loss>{:5.5f} Val.loss>{:5.5f}".format(j, train_perf, val_accuracy, train_loss, val_loss))

		elapsed_session_time = (time.time() - start_session_time) / 60.0
		print("Time Passed: {:7.2f}min".format(elapsed_session_time))

		comp_iters = (j+1)*TRAIN_ITERS_PER_EPOCH
		if comp_iters == 24000:
			learning_rate = learning_rate / learning_rate_decay_factor
			print("EPOCH---{} Reducing learning rate...".format(j))
		if comp_iters > 24000 and (comp_iters - 24000) % 8000 == 0:
			learning_rate = learning_rate / learning_rate_decay_factor
			print("EPOCH---{} Reducing learning rate...".format(j))

		if val_accuracy > best_val_accuracy:
			best_val_accuracy = val_accuracy
			print("best val accuracy -> saving parameters to best-model.ckpt")
			saver.save(sess, MODEL_PATH + "best-model.ckpt")

	sess.close()

else:

	with tf.Session() as sess:

		saver = tf.train.Saver()
		saver.restore(sess, tf.train.latest_checkpoint(MODEL_PATH))

		validation_handle = sess.run(validation_iterator.string_handle(), feed_dict={batch_size: BATCH_SIZE})
		sess.run(validation_iterator.initializer, feed_dict={batch_size: BATCH_SIZE})
		print("Running best model on Validation samples")
		val_acc = np.float32(0.0)
		kk = 0
		cnt2 = 0
		while True:
			try:
				curr_acc_val = sess.run(curr_count, feed_dict={batch_size: BATCH_SIZE, is_training: False, handle: validation_handle})
				val_acc += curr_acc_val
				kk += 1
				#if kk % 12800 == 0:
					#cnt2+=1
					#print("{:2.2f}--> %age BATCHESx100 done".format(cnt2*12800/tot_val_samples))
			except tf.errors.OutOfRangeError:
				print("Number of Val. samples-->{}".format(kk))
				val_accuracy = (val_acc / (kk * BATCH_SIZE)) * 100
				#print(val_accuracy)
				print("Validation accuracy on {} samples is --->{:5.5f}".format(kk*BATCH_SIZE, val_accuracy))
				break
