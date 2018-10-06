import os, numpy as np
import constants as cs
import pickle

try:
	import cv2
except:
	import sys
	sys.path.append('/usr/local/lib/python2.7/site-packages')
	import cv2

def remove_augmented_data(folders):
	for folder in folders:
		all_images=os.listdir(folder)
		for image_name in all_images:
			img_path=folder+'/'+image_name
			for suffix in cs.suffixes.values():
				if suffix in img_path:
					try:
						os.remove(img_path)
					except:
						pass

def augment_and_save(folders):
	for folder in folders:
		all_images=os.listdir(folder)
		for image_name in all_images:
			img_path=folder+'/'+image_name
			image=cv2.imread(img_path,cv2.IMREAD_COLOR)
			if image is None:
				print('Could not read file: ', img_path)
				continue
			else:

				transform1=cv2.GaussianBlur(image,(3,3),sigmaX=0.2,sigmaY=0.2) #add gaussian blur to image
				cv2.imwrite(img_path.replace('.jpg',cs.suffixes['gb']),transform1)  #image with gaussian blur

				if np.random.random() <0.5: #we either flip the image or get a perspective transform about some points
					transform2=np.fliplr(image)
					transform3= cv2.GaussianBlur(transform2,(3,3),sigmaX=0.2,sigmaY=0.2) #add gaussian blur to transform2
					cv2.imwrite(img_path.replace('.jpg',cs.suffixes['f']),transform2)  #image flipped
					cv2.imwrite(img_path.replace('.jpg',cs.suffixes['fgb']),transform3)   #flipped and gaussian blur
				else:
					transform2 = cv2.warpPerspective(image,cs.perspective_matrix,(cs.im_width,cs.im_height))
					transform3= cv2.GaussianBlur(transform2,(3,3),sigmaX=0.2,sigmaY=0.2) #add gaussian blur to transform2
					cv2.imwrite(img_path.replace('.jpg',cs.suffixes['p']),transform2)   #image perspective transform
					cv2.imwrite(img_path.replace('.jpg',cs.suffixes['pgb']),transform3) #transform with gaussian blur
				
				#implements unsharp masking
				mask= cv2.filter2D(image,-1,cs.unsharp_kernel)
				transform4= cv2.addWeighted(src1=image,alpha=1.05,src2=mask,beta=-0.05,gamma=0) #output= alpha*src1 + beta+src2 + gamma
				cv2.imwrite(img_path.replace('.jpg',cs.suffixes['um']),transform4) #unsharp masked image

				if cs.allow_rotation:
					#implements rotation
					rows,cols = cs.im_width,cs.im_height
					angle=cs.min_angle+np.random.random()*(cs.max_angle-cs.min_angle)
					M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
					transform5= cv2.warpAffine(image,M1,(cols,rows))
					cv2.imwrite(img_path.replace('.jpg',cs.suffixes['r']),transform5) #rotated image

				break
				#print('Done')

def make_batches(folders,num_batches=5, valid=0.1,test=0.1):
	train_features=[]
	train_predicts=[]

	valid_features=[]
	valid_predicts=[]

	test_features =[]
	test_predicts =[]

	for folder in folders:
		all_images=np.array(os.listdir(folder))
		#print len(all_images)
		#indices=np.random.shuffle(np.arange(len(all_images)))
		#print indices
		np.random.shuffle(all_images)
		images_in_folder=[]
		
		for image_name in all_images:
			img_path=folder+'/'+image_name
			image=cv2.imread(img_path,cv2.IMREAD_COLOR)
			
			if image is None:
				print('Could not read file: ', img_path)
				continue
			else:
				image=image/255.0 #normalized
				if image.shape == (cs.im_width,cs.im_height,3):
					images_in_folder.append(image)

		train_features.extend(images_in_folder[:int(all_images.shape[0]*(1-valid-test))])
		valid_features.extend(images_in_folder[int(all_images.shape[0]*(1-valid-test)):int(all_images.shape[0]*(1-test))])
		test_features.extend(images_in_folder[int(all_images.shape[0]*(1-test)):])
		
		predict=[np.eye(cs.numdays)[int(folder[folder.find('_')+1:])]]
		print(predict)
		#predict=[np.float32(folder[-1])]
		all_predicts=predict*all_images.shape[0]

		train_predicts.extend(all_predicts[:int(all_images.shape[0]*(1-valid-test))])
		valid_predicts.extend(all_predicts[int(all_images.shape[0]*(1-valid-test)):int(all_images.shape[0]*(1-test))])
		test_predicts.extend(all_predicts[int(all_images.shape[0]*(1-test)):])

		#break #uncomment when testing. This is VERY important.
	
	train_features=np.array(train_features).reshape(len(train_features),32,32,3)
	train_predicts=np.array(train_predicts)

	indices=np.arange(train_features.shape[0])
	np.random.shuffle(indices)
	
	train_features=train_features[indices,:,:,:]
	train_predicts=train_predicts[indices]
	
	batch_size=int(train_features.shape[0]/num_batches)

	for batch_num in range(1,num_batches+1):
		if batch_num<batch_size:
			batch_features=train_features[(batch_num-1)*batch_size:batch_num*batch_size,:,:,:]
			batch_predicts=train_predicts[(batch_num-1)*batch_size:batch_num*batch_size]
		else:
			batch_features=train_features[(batch_num-1)*batch_size:,:,:,:]
			batch_predicts=train_predicts[(batch_num-1)*batch_size:]

		pickle.dump((batch_features, batch_predicts), open('batch_'+str(batch_num)+'.p', 'wb')) #+'.p'
		print('Wrote {} features and {} predictions in batch #{}'.format(batch_features.shape[0],batch_predicts.shape[0], batch_num))

	valid_features=np.array(valid_features).reshape(len(valid_features),32,32,3)
	valid_predicts=np.array(valid_predicts)
	#print valid_features.shape
	indices=np.arange(valid_features.shape[0]) #for safety
	np.random.shuffle(indices)
	valid_features=valid_features[indices,:,:,:]
	valid_predicts=valid_predicts[indices]
	pickle.dump((valid_features, valid_predicts), open('validation.p', 'wb')) #.p
	print('Wrote {} features and {} predictions in validation set'.format(valid_features.shape[0],valid_predicts.shape[0]))

	test_features=np.array(test_features).reshape(len(test_features),32,32,3)
	test_predicts=np.array(test_predicts)
	indices=np.arange(test_features.shape[0])-2
	np.random.shuffle(indices)
	test_features=test_features[indices,:,:,:]
	test_predicts=test_predicts[indices]
	pickle.dump((test_features, test_predicts), open('test.p', 'wb')) #+'.p'
	print('Wrote {} features and {} predictions in test set'.format(test_features.shape[0],test_predicts.shape[0]))


def load_batch(batch_num):
	features, predictions =pickle.load(open('batch_'+str(batch_num)+'.p' ,mode='rb')) #mode='rb'
	return features, predictions

def load_valid():
	features, predictions =pickle.load(open('validation.p', mode='rb')) #mode='rb',
	return features, predictions

def load_test():
	features, predictions =pickle.load(open('test.p',mode='rb')) # mode='rb',
	return features, predictions