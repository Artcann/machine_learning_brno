# Explanation of my Neural Network

I started by using classic fit network, but didn't succeed in achieving more than 60% of accuracy on the test model. I did some research and found that Convolutional Neural Network works best for images.

I tried the CNN approach, and had better results, with approx 70% without tuning. I added 2 layers of Conv2D and implemented Keras-Tuning to tune the Hyperparameters of the NN. With this I achieved 95% accuracy on the training set, but only 80% on the testing set.

I had an overfitting problem on the training dataset, so I added a dropout layer to try and correct, with the final result being an accuracy of 91% and a loss of 29% on the testing set. I could probably tune even more the Network to achieve 95%, but I'm pretty happy with this result and the performance of my GPU in computing it.
