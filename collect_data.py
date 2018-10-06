try:
	import cv2
except:
	import sys
	sys.path.append('/usr/local/lib/python2.7/site-packages')
	import cv2

import os, numpy as np

cap=cv2.VideoCapture(0)
classifier=cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
scale=4

def find_last_num(folder):
	files=os.listdir(folder)
	return len(files) #no need to add +1 since we start counting from 0
						#so if last saved is 74, len is 75

def get_data():
	day_num=raw_input('Please enter days since last shaved: ')
	folder='day_'+str(day_num)
	count=0
	ret,frame=cap.read()

	if os.path.exists(folder):
		pass
	#find last number of images saved
		last_num=find_last_num(folder) #last saved image +1
		
	else:
		os.makedirs(folder)
		last_num=0 # start counting from 0

	while count<500:

		gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		
		faces=classifier.detectMultiScale(gray_frame[::scale,::scale],1.1,5)

		if len(faces)==0:
			continue
		else:
			(x,y,w,h)=list(scale*np.array(faces[0])) #we expect only 1 face in the frame
			#print(x,y,w,h)
			if w>100 and h>100:
				cropped=cv2.resize(frame[y:y+h+20,x+25:x+w-25,:],(32,80))
				cropped=cropped[48:,:,:]
				cv2.imwrite(folder+'/'+str(last_num)+'.jpg',cropped)
				last_num+=1
				count+=1
		for face in faces:
			#(x,y,w,h)=list(scale*np.array(face))
			cv2.rectangle(frame,(x-10,y-10),(x+w+10,y+h+10),(0,255,0),2)
		cv2.imshow('frame',frame)
		if(cv2.waitKey(1)== ord("q")):
			break
		
		ret,frame=cap.read()

	cv2.destroyAllWindows()
	cap.release()


	print('Finished capturing images')

if __name__ == '__main__':
	get_data()