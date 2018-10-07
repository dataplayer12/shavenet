import os

#image parameters
im_width=32
im_height=32

##parameters for training
num_batches = 10
epochs=5
learning_rate=1e-3
dropout=0.75 #probability of keeping neurons during dropout
all_days=[int(f[4:]) for f in os.listdir('./data') if f.startswith('day')]
numdays=max(all_days)-min(all_days)+1 #0 to 7