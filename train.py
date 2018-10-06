import os, numpy as np
import data_utils as du
import tensorflow as tf
import constants as cs

#Step 1. Augment images and store them in the same folder : can be done now
#Step 2. Split all data into train, validation and test : can be done now
##train is further split into batches
#Step 3. Build neural net
## Architecture: inspired by alexnet
	## input layer: 32x32x3 image
	## conv1 layer 3x3x3x8 (zero padded,relu) output: 32x32x8
	## conv2 layer 3x3x8x16 (zero padded,relu) output: 32x32x16
	## pool1 layer 2x2 output output: 16x16x16

	## conv3 layer 3x3x16x32 (zero padded,relu) output: 16x16x32
	## pool2 layer 2x2 output: 8x8x32
	## pool3 layer 2x2 output: 4x4x32
	## flatten 4x4x32 -> 1x512
	## fully connected1 layer: 4 relu output: 4 (512x4)
	## output layer: 1 output neuron, no activation output: 1 (4x1)
	## cost is mean-squared error
	## this results in a total of 8089 parameters, 8028 weights and 61 biases

#Step 4. load batches : can be done now
## Load in images
## Get predictions y

#Step 5. train model
#Step 6. save model with tensorflow

##Write predict script to predict on camera feed

folders=[os.path.join('data/'+isDir) for isDir in os.listdir('data') if os.path.isdir('data/'+isDir)]
#print(folders)

#du.augment_and_save(folders) #WARNING: DO THIS ONLY ONCE, OR YOU WILL END UP WITH A VERY LARGE DATASET
#du.make_batches(folders,cs.num_batches) #DO ONLY ONCE

def get_mini_batch(x,y,mb_size):
	for idx in range(0,y.shape[0],mb_size):
		mb_x=x[idx:idx+mb_size,...]
		mb_y=y[idx:idx+mb_size,:]
		yield mb_x,mb_y

#define computational graph
if len(os.listdir('checkpoints'))<3:
	restore_pretrained_model=False
	weights={'conv1': tf.Variable(tf.truncated_normal([3,3,3,8],stddev=1/50.0),name='w_c1'),
			 'conv2': tf.Variable(tf.truncated_normal([3,3,8,16],stddev=1/50.0),name='w_c2'),
			 'conv3': tf.Variable(tf.truncated_normal([3,3,16,32],stddev=1/50.0),name='w_c3'),
			 'conv4': tf.Variable(tf.truncated_normal([3,3,32,64],stddev=1/50.0),name='w_c4'),
			   'fc1': tf.Variable(tf.truncated_normal([1024,64],stddev=1/50.0),name='w_fc1'),
			   'out': tf.Variable(tf.truncated_normal([64,cs.numdays],stddev=1/50.0),name='w_o')}

	biases={'conv1': tf.Variable(tf.zeros([8]),name='b_c1'),
			'conv2': tf.Variable(tf.zeros([16]),name='b_c2'),
			'conv3': tf.Variable(tf.zeros([32]),name='b_c3'),
			'conv4': tf.Variable(tf.zeros([64]),name='b_c4'),
			  'fc1': tf.Variable(tf.zeros([64]),name='b_fc1'),
			  'out': tf.Variable(tf.zeros([cs.numdays]),name='b_o')}

	with tf.name_scope('inputs'):
		inputs_ = tf.placeholder(tf.float32,[None,32,32,3],name='input_x')
		y_ = tf.placeholder(tf.float32,[None,cs.numdays],name='input_y')
		pkeep=tf.placeholder(tf.float32,name='dropout')
		learning_rate=tf.placeholder(tf.float32,name='lr')

	with tf.name_scope('conv1'):
		conv1= tf.nn.conv2d(inputs_,weights['conv1'],strides=[1,1,1,1],padding='SAME')
		conv1= tf.nn.bias_add(conv1, biases['conv1'])
		conv1= tf.nn.relu(conv1)
		conv1= tf.nn.dropout(conv1,keep_prob=pkeep)

	with tf.name_scope('conv2'):
		conv2= tf.nn.conv2d(conv1,weights['conv2'],strides=[1,2,2,1],padding='SAME')
		conv2= tf.nn.bias_add(conv2,biases['conv2'])
		conv2= tf.nn.relu(conv2)

	with tf.name_scope('conv3'):
		conv3= tf.nn.conv2d(conv2,weights['conv3'],strides=[1,2,2,1],padding='SAME')
		conv3= tf.nn.bias_add(conv3, biases['conv3'])
		conv3= tf.nn.relu(conv3)
		conv3= tf.nn.dropout(conv3, keep_prob= pkeep)


	with tf.name_scope('conv4'):
		conv4= tf.nn.conv2d(conv3,weights['conv4'],strides=[1,2,2,1],padding='SAME')
		conv4= tf.nn.bias_add(conv4, biases['conv4'])
		conv4= tf.nn.relu(conv4)
		conv4= tf.nn.dropout(conv4, keep_prob= pkeep)

	with tf.name_scope('flatten'):
		flatten=tf.reshape(conv4,[-1,1024])

	with tf.name_scope('fc1'):
		fc1= tf.add(tf.matmul(flatten,weights['fc1']),biases['fc1'])

	with tf.name_scope('output'):
		out= tf.add(tf.matmul(fc1,weights['out']),biases['out'])
		out= tf.nn.softmax(out)
	with tf.name_scope('cost'):
		cost= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out,labels=y_),name="cost_value")

	with tf.name_scope('optimizer'):
		optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate,name="my_optimizer").minimize(cost)

else:
	restore_pretrained_model=True


with tf.Session() as sess:
	if not restore_pretrained_model: #starting from scratch
		saver = tf.train.Saver() #initialize saver
		sess.run(tf.global_variables_initializer())
		tf.add_to_collection('train_op', optimizer) #add optimizer to collection so that it can be restored later
		writer = tf.summary.FileWriter('./graphs', sess.graph) #write graph
		valid_x,valid_y= du.load_valid() #load validation set
		for epoch in range(cs.epochs):
			for batch_num in range(1,cs.num_batches+1):
				batch_x,batch_y=du.load_batch(batch_num=batch_num)
				for mb_x,mb_y in get_mini_batch(batch_x,batch_y,64):
					lr=2e-2/batch_num #learning rate
					feeds={inputs_:mb_x,y_:mb_y,pkeep: cs.dropout,learning_rate: lr}
					train_loss,_ = sess.run([cost, optimizer],feed_dict=feeds)
					print("Epoch: {}/{}, batch {}...".format(epoch+1, cs.epochs,batch_num),
		              "Training loss: {:.4f}".format(train_loss))
				if batch_num%5==0:
					#train_loss=sess.run(cost,feed_dict=feeds)
					valid_loss=sess.run(cost,feed_dict={inputs_: valid_x, y_: valid_y, pkeep: 1.0})
					print("Epoch: {}/{}, batch {}...".format(epoch+1, cs.epochs,batch_num),
	                "Validation loss: {:.4f}".format(valid_loss))
		saver.save(sess,"checkpoints/epoch",global_step=cs.epochs)
		writer.close()

	else: #restore a previously trained model and train further
		last_global_step=max([int(filename[6]) for filename in os.listdir('checkpoints') if 'meta' in filename])
		saver=tf.train.import_meta_graph('checkpoints/epoch-'+str(last_global_step)+'.meta') #graph does not change
		#print(last_global_step)
		saver.restore(sess,tf.train.latest_checkpoint('./checkpoints/'))
		print('Found pretrained model. Loaded latest checkpoint')
		graph=tf.get_default_graph()
		inputs_=graph.get_tensor_by_name("inputs/input_x:0")
		y_= graph.get_tensor_by_name("inputs/input_y:0")
		pkeep=graph.get_tensor_by_name("inputs/dropout:0")
		learning_rate=graph.get_tensor_by_name("inputs/lr:0")
		cost=graph.get_tensor_by_name("cost/cost_value:0")
		#optimizer=graph.get_tensor_by_name("optimizer/Adam")
		#optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate,name="my_optimizer").minimize(cost)
		#sess.run(tf.global_variables_initializer())
		optimizer=tf.get_collection('train_op')[0]
		valid_x,valid_y= du.load_valid()
		
		for epoch in range(cs.epochs):
			for batch_num in range(1,cs.num_batches+1):
				batch_x,batch_y=du.load_batch(batch_num=batch_num)
				lr=cs.learning_rate/epoch
				for mb_x,mb_y in get_mini_batch(batch_x,batch_y,64):

					feeds={inputs_:mb_x,y_:mb_y,pkeep: cs.dropout,learning_rate: lr}
					train_loss,_ = sess.run([cost,optimizer],feed_dict=feeds)
					print("Epoch: {}/{}, batch {}...".format(epoch+1, cs.epochs,batch_num),
		              "Training loss: {:.4f}".format(train_loss))
				if batch_num%5==0:
					#train_loss=sess.run(cost,feed_dict=feeds)
					valid_loss=sess.run(cost,feed_dict={inputs_: valid_x, y_: valid_y, pkeep: 1.0})
					print("Epoch: {}/{}, batch {}...".format(epoch+1, cs.epochs,batch_num),
	                "Validation loss: {:.4f}".format(valid_loss))
		saver.save(sess,"checkpoints/epoch",global_step=last_global_step+cs.epochs)