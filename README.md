**Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/center.jpg "Center Driving"
[image3]: ./examples/left_recover.jpg "Recovery Image"
[image4]: ./examples/right_recover.jpg "Recovery Image"
[image5]: ./examples/right_recover2.jpg "Recovery Image"
[image6]: ./examples/center.jpg "Normal Image"
[image7]: ./examples/center_flip.jpg "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths of 6 (model.py lines 66-84) 

The model includes ELU layers after each Convolution and Fully connected layers except the output layer to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 68). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce over-fitting (model.py lines 89, 92, 95). The models architecture was adjusted several times to address over-fitting. Also attempted to use batch normalization between the convolution layers.

The model was trained and validated on different data sets to ensure that the model was not over-fitting in the keras Sequence model compile function (code line 103). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 109). The number of epoch, batch size and steering correction factors were tuned. The steering correction factor was tunned to compensate for the models tendency to pull to inside of turns. The exclusion angle was tuned to compensate for the models bias for going straight.

####4. Appropriate training data

Training data was created to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to start trying multiple architectures which included Lenet, Alexnet and the Nvidia. The most suitable selected and adjusted as required to reduce loss and minimize over-fitting.

My first step was to try multiple architectures and assess their initial results, loss and the extent of over-fitting. The lenet architecture seemed to be the least problematic.

I was able successfully use the Lenet architecture on the previous project to classify traffic signs and was not as complicated as the Nvidia architecture.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that the model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was over-fitting. 

To combat the over-fitting, I iteratively modified the model a convolution and fully connected layer, reduce the filter depth on the remaining convolution layers and included dropout on the fully connected layers.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. I collected recovery data at those spot to improve the driving behavior in these cases. Some corners were very problematic and recovery data didn't seem to help so collected data of the car taking those turns perfectly one time and added it to the data set repeatedly until the car could take the turn perfectly

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 66-99) consisted is illustrated in the table below:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 64x64x3 raw pixel values of full color image  | 
| Normalization         | 64x64x3 raw pixel values of full color image  | 
| Convolution 5x5     	| 1x1 stride, same padding, 6 depth         	|
| ELU					| Element wise activation thresholds at zero.   |
| Max pooling	      	| 2x2 pool size             	                |
| Convolution 5x5	    | 1x1 stride, same padding, 6 depth         	|
| ELU					| Element wise activation thresholds at zero.   |
| Max pooling	      	| 2x2 pool size                                 |
| Convolution 3x3     	| 1x1 stride, same padding, 6 depth         	|
| ELU					| Element wise activation thresholds at zero.   |
| Max pooling	      	| 2x2 pool size                                 |
| Convolution 3x3	    | 1x1 stride, same padding, 6 depth         	|
| ELU					| Element wise activation thresholds at zero.   |
| Max pooling	      	| 2x2 pool size                                	|
| Flatten       		| Flatten the input from the convolution layers |
| Fully connected		| A fully connected layer with 100 units.       |
| ELU					| Element wise activation thresholds at zero.   |
| Dropout				| keep probability 0.2,to minimize over fitting.|
| Fully connected		| A fully connected layer with 50 units.        |
| ELU					| Element wise activation thresholds at zero.   |
| Dropout				| keep probability 0.2,to minimize over fitting.|
| Fully connected		| A fully connected layer with 10 units.        |
| ELU					| Element wise activation thresholds at zero.   |
| Dropout				| keep probability 0.5,to minimize over fitting.|
| Fully connected		| A fully connected layer with 1 units.         |
|						|												|


####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover if it begins to run off the road. These images show what a recovery looks like starting from right or left side of the road:

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process with car going in the other direction in order to get more data points.

To augment the data set, I also flipped images and angles thinking that this would help the model generalize better. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

After the collection process and preprocessing, I had 339,868 data points. The data was preprocessed by cropping the unneeded upper and lower portions of the images , resizing the to 64x64,  created flipped duplicates. Note. Images from the side cameras were also used with steering measurements computed to encourage the car to avoid pull to the sides of the road.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as evidenced by no further improvements in validation loss was apparent. I used an adam optimizer so that manually training the learning rate wasn't necessary.
