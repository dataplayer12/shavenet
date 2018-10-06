import os
import numpy as np

#image parameters
im_width=32
im_height=32

#allow rotation from here
allow_rotation=False
if allow_rotation:
	min_angle=-5
	max_angle= 5

#kernel for unsharp masking
unsharp_kernel=np.array([[-1, -1 ,-1],[-1, 8, -1],[-1, -1, -1]])

#kernel for perspective transform
pts1 = np.float32([[2,2],[im_width-2,2],[2,im_height-2],[im_width-2,im_height-2]])
pts2 = np.float32([[0,0],[im_width,0],[0,im_height],[im_width,im_height]])
perspective_matrix=np.array([[  1.14285714e+00,   1.38777878e-17,  -2.28571429e+00],
 [ -2.08166817e-16,   1.14285714e+00,  -2.28571429e+00],
 [ -3.46944695e-17,  -1.30104261e-18,   1.00000000e+00]])
#perspective_matrix = cv2.getPerspectiveTransform(pts1,pts2)
#print perspective_matrix
#augmented data are stored with these suffixes in their name
#gb: gaussian blur, f: flip, fgb: flip with gb, p: perspective,
#pgb: perspective with gb, um: unsharp masked, r: rotated

suffixes={'gb':'_gb.jpg','f':'_f.jpg','fgb':'_fgb.jpg','p':'_p.jpg','pgb':'_pgb.jpg','um':'_um.jpg','r':'_r.jpg'}

##parameters for training
num_batches = 10
epochs=5
learning_rate=1e-3
dropout=0.75 #probability of keeping neurons during dropout
all_days=[int(f[4:]) for f in os.listdir('./data') if f.startswith('day')]
numdays=max(all_days)-min(all_days)+1 #0 to 7