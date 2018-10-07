# ShaveNet

## Do you forget to shave your face when you are busy programming?

This is a light-hearted project which I conceived of when shaving my beard.
I often skip shaving until I realize I look like a chimpanzee.

This project was conceived to remind myself when to shave. Since there are no existing labelled datasets for male faces showing days since last shaved, I created a script which allows one to generate their own labelled data.

### Bro, did you just assume my gender?
No, I didn't. I am not sexist or *beardist*. If you do not sport a beard, fear not! By making a few changes, mainly to data collection and test scripts, you can adapt ShaveNet to recognize emotions. You can save data with different emotions as labels and train your own emotion detector! If you happen to be Kristen Stewart, you can't be helped.

### Who is this for?
The code in this repo is a good guide for a beginner to neural networks and tensorflow. The scripts are a good tutorial for an end to end machine learning project involving data *collection*, *augmentation*, low level *implementation* of a ConvNet in tensorflow with `tf.nn` API, evaluation of training with *tensorboard* and *evaluation* on a live camera feed.
From a practitioner's point of view, it is more satisfying to see how machine learning works on your own face data, rather than arbitrary objects from a well known dataset.

### How to use?

Here are the four basic steps to use this repo:

Step 1. Shave your beard on day `0`. From day `0` to about one week, record training data using `collect_data.py`.

Step 2. Apply data augmentation with [hey-daug](https://github.com/dataplayer12/hey-daug).

Step 3. Train a fairly shallow ConvNet with `train.py`.

Step 4. Test on your face with `test.py`.

### Results
Yes, it has been 6 days!

![alt text][train]
*Fake smile for the internet*

[train]: https://github.com/dataplayer12/shavenet/raw/master/results.png "Results on my face."


### Caution
Most of the tensorflow syntax is very old (written for tf 1.1) and low-level. This is nowhere near an optimal architecture for solving the problem. The basic demonstration is complete, but this is still a work in progress.

### Next steps
Train a model to advise when to bathe.
