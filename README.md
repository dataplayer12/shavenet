# shave

## Do you forget to shave your face when you are busy programming?

This is a light-hearted project which I conceived of when shaving one day.
I often skip shaving until I realize I look like an alien.

This project was conceived to remind myself when to shave. Since there are no existing labelled datasets for male faces showing days since last shaved, I created a script which allows one to generate their own labelled data.

Here are the three basic steps to use this repo:

Step 1. Shave on day `0`. From day `0` to about one week, record training data using `collect_data.py`.

Step 2. Train a fairly shallow ConvNet with `train.py`.

Step 3. Test on your face with `test.py`.

Yes, it has been 6 days!

![alt text][train]

[train]: https://github.com/dataplayer12/shave/raw/master/results.png "Results on my face."


### Caution
Most of the tensorflow syntax is very old (written for tf 1.1) and low-level. This is nowhere near an optimal architecture for solving the problem. The basic demonstration is complete, but this is still a work in progress.

### Future Work
Train a model to advise when to bathe.
Have it bathe automatically.