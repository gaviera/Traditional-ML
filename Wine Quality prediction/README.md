# Wine Quality - Linear Regression
My first ML project. A simple linear regression with 11 features.

## How it's work?

First of all, i'm using data from [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/dataset/186/wine+quality)
The data was splitted 80/20 to train and test.

I used Mean Square Error and gradient descent.

After training my ML with 5000 EPOCHS, i got these results:

My LOSS curve during the training:

![image](https://github.com/gaviera/Traditional-ML/assets/47997823/3a67efb0-3d07-4088-b224-d89bebfffd9d)

This graph show's the training data and predictions with test data:

![image](https://github.com/gaviera/Traditional-ML/assets/47997823/fd566041-7813-4e40-8394-8e885bbfa67d)

As we can see, the training data are integers from 0 to 10, so problably that's why i got kinda low predictions...


And finally i test to do 15000 EPOCHS (3 times more),

the LOSS Curve:

![image](https://github.com/gaviera/Traditional-ML/assets/47997823/80bba6fd-f5b2-44f5-a8ed-f4458a23f38c)

as we can see, the loss converges to 0.65 approximately, so do more EPOCHS will be a waste of time (and resouces)

With the test data i got a 0.76 of loss and i got the followings weights (to anyone who wants to test)

[[-1.10969927e-02], [ 6.16047856e-01], [ 1.33063320e-02], [ 1.60487429e-02], [ 1.35238199e-01], [ 8.97172904e-03], [ 1.20174922e-04], [ 3.16538202e-01], [ 2.93237849e-01], [ 5.10572140e-01], [ 3.60638249e-01]]
