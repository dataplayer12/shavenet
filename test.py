from __future__ import print_function
import os, numpy as np
import data_utils as du
import tensorflow as tf
import constants as cs
import time

try:
	import cv2
except:
	import sys
	sys.path.append('/usr/local/lib/python2.7/site-packages')
	import cv2

while True:
	try:
		live_test=int(raw_input("Please enter 1 to test live and 0 to test on validation set: "))
		break
	except:
		print("Unidentified input. Please try again")
		pass

with tf.Session() as sess:
	last_global_step=max([int(filename[6]) for filename in os.listdir('checkpoints') if 'meta' in filename])
	saver=tf.train.import_meta_graph('checkpoints/epoch-'+str(last_global_step)+'.meta') #graph does not change
	saver.restore(sess,tf.train.latest_checkpoint('./checkpoints/'))
	print('Found pretrained model. Loaded latest checkpoint')
	graph=tf.get_default_graph()
	inputs_=graph.get_tensor_by_name("inputs/input_x:0")
	pkeep=graph.get_tensor_by_name("inputs/dropout:0")
	out=graph.get_tensor_by_name("output/Add:0")	
	#print(np.histogram(np.array(valid_y),bins=np.arange(10)))
	
	if not live_test:
		valid_x,valid_y= du.load_valid()
		while True:
			to_continue=raw_input("Please enter any key to continue and 'x' to stop:")
			if to_continue=='x':
				break
			else:
				index=np.random.randint(low=0,high=valid_y.shape[0])
				example=valid_x[index,:,:,:]
				ex_label=np.argmax(valid_y[index])
				prediction= sess.run(out,feed_dict={inputs_:example[np.newaxis,:,:,:], pkeep: 1.0})
				print(prediction[0])
				prediction=np.argmax(prediction[0])
				print("True value for image was {} and prediction was {}".format(ex_label,prediction))

	else:
		count=0
		while count<100:
			cap=cv2.VideoCapture(0)
			time.sleep(5)
			classifier=cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
			scale=4
			ret,frame=cap.read()
			gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
			faces=classifier.detectMultiScale(gray_frame[::scale,::scale],1.1,5)

			if len(faces)==0:
				continue
			else:
				(x,y,w,h)=list(scale*np.array(faces[0])) #we expect only 1 face in the frame
				
				if w>100 and h>100:
					cropped=cv2.resize(frame[y:y+h+20,x+25:x+w-25,:],(32,80))
					example=cropped[48:,:,:]
					cv2.rectangle(frame,(x-10,y-10),(x+w+10,y+h+10),(0,255,0),2)
					prediction=sess.run(out,feed_dict={inputs_:example[np.newaxis,:,:,:],pkeep: 1.0})
					print(prediction[0])
					prediction=np.argmax(prediction[0])
					#print(prediction)
					text_msg="I think you last shaved {} days ago".format(prediction)
					cv2.putText(frame,text_msg, (x,y-20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
					count+=1
					
			cv2.imshow('frame',frame)
			if(cv2.waitKey(1)== ord("q")):
				break
			
			ret,frame=cap.read()

		cv2.destroyAllWindows()
		cap.release()