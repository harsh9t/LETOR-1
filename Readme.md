========================================================
 Search Parameter Tuning on Microsoft Learning to Rank (LETOR) 4.0 by Linear Regression
========================================================


 OVERVIEW
----------
Linear regression is implemented on web search ranking data set LETOR 4.0. We are given a dataset of vectors and target values. There are 46 features for each sample in the dataset. We need to learn the weight of each feature which can give us the best result given a new sample. We use the technique of regression to learn these weights by using the samples given. We use the Gaussian Model of regression and vary the degree or complexity of the model until it gives us the closest prediction we can get on a previously unseen keyword.

 OBJECTIVE
-----------
Given: a set of training examples (xn, tn), n =1 to N, where N is the total number of samples
Goal: learn a function y(x) to minimize a loss function (error function): E(y,t)

 REGRESSION MODEL
 -----------------
A Regression Model is first fed some sample data, to proceed by arriving at a tentative set of values of the weights. This is called the training set. The training set should contain a substantial number of samples for fine tuning the weights of the regression model. The more the samples, the better an estimate do we get about what the values of the weights will be. This set of values are evaluated for accuracy against a smaller subset of the data called the validation set, whose purpose is to test how well is the model performing with its current complexity or degree. This is a small set which is just used to evaluate the error the model is generating on unlearned data. We chose that degree of the model which gives us the minimum error, and we designate that as the most accurate complexity of the model, which gives us the closest prediction given an unlearned value, from the rest of the dataset which we call the testing set.
For this purpose, we partition the dataset into 3 parts containing 40%, 10% and 50% of the total values respectively. We use the 40% partition as the training set, 10% as a validation set and the remaining 50% as the validation set. We separate out the first column as it has the target values for all the samples, we refer to as target vector t.
With the data sets ready, we proceed to apply the technique of regression on training set. Regression is a general way of obtaining a linear model that fits a given set of data points as accurately as possible. This estimation is subject to error from the actual values, which we use to train the model to improve its accuracy and provide a general model which minimizes the error across all the given data values.

We use the linear basis function model which is given by:
              y(x,w) = w0 + j=1∑M-1 wj φj(x) = φj(x)w

Where M is the degree of the model
w0, w1 … to wM-1 are the weights we need to calculate
φ is the basis function we use to calculate w.

In the simplest case φj(x) = xj. But we take the value of φ from Gaussian distribution as
                φ(x) = exp{ - (x - µ)¬¬2/2s2}       where µ is the mean and s is the variance

µ and s are called the hyper-parameters of the model with degree M. For a model with degree M there are M basis functions. Typically φ0(x) is 1, so w0 acts as a bias parameter.
We calculate µ and s from the data in the training set and prepare the design matrix for the data by substituting each data value in each basis-function successively. Hence, a model of degree M will have a design matrix of 1+(M-1)×D columns long, where D = 46 in our case. We get each set of 46 columns from each of the basis-functions φ1, φ2, … ,φM respectively.


Hence the design matrix φ is

     1 φ1(x11) φ1(x12) … φ1(x1D) φ2(x11) φ2(x12) … φ2(x1D) … φM(x11) φM(x12) … φM(x1D)
     1 φ1(x21) φ1(x22) … φ1(x2D) φ2(x21) φ2(x22) … φ2(x2D) … φM(x21) φM(x22) … φM(x2D)

Φ =       ⁞          ⁞          ⁞          ⁞          ⁞          ⁞          ⁞

     1 φ1(xN-11)  φ1(xN-12) … φ1(xN-1D) φ2(xN-11) φ2(xN-12) … φ2(xN-1D) … φM(xN-11) φM(xN-12) … φM(xN-1D)
     1 φM(xN1)    φ1(xN2)   … φ1(xND)   φ2(xN1)   φ2(xN2)   … φ2(xND)   … φM(xN1)   φM(xN2)  …  φM(xND)

Using the design matrix Φ, we can calculate the weights by

    w* = (ΦT Φ)-1 ΦTt       where t is the target vector

On calculating the values for each sample given by the model, we can find out the error between the model’s values and the actual values that are given in the data set. We measure this error by calculating the root means square error or ERMS given as

        ERMS(w) = [(ΦTw – t)( Φw – t)/N]1/2

To prevent against a higher degree we use a regularization parameter λ to limit the increase in M, due to which the model can become quite complex. We introduce λ in our equation for w and thus into ERMS as

        w* = (ΦT Φ + λI)-1 ΦTt      where I is an identity matrix

Small weight values of λ are encouraged.

For every value of M and λ, we obtain the basis functions φ1, φ2, … ,φM, Φ and w* from the training set and proceed to validate our model on the validating set. With increasing M the RMS error value on the validation set denoted as Evalidation, decreased, but we were facing the problem of over-fitting, so after introducing λ to small value the Evalidation decreased further, on a lesser value of M, thus making the model simpler while decreasing the error at the same time.

If we keep increasing the M, after considering the regularization parameter, the Evalidation begins to increase after a certain limit. This indicates that an optimal value of M and λ was reached in the previous iteration. Hence, we arrive at the optimal complexity for the model. We go further to show how this model performs on the testing set and represent the actual performance through Etest.

A graph plot comparing E¬test and Etraining represents the choice of the optimal value of M, also depicting the variation of error for each value of M.


 RESULTS
---------
The degree of the model was increased incrementally from 1 through 30, which sufficiently showed the optimal value of M required.
The regularization factor λ was varied beginning from 0.001 to 1000 in multiples of 10, but a lower error and thus the optimal performance of the model was found to be at λ = 0.001 and M = 23.

Minimization of Error:
As it is evident from comparing the two plots for different values of λ, we have got lesser Etesting for lower values of λ. After determining an optimal weight value for λ, we see that on increasing the complexity or the degree of the model, the error keeps decreasing, and then becomes almost constant and starts rising again. At that point, it shows that the particular value of M is the appropriate degree of our model. We also find that EVALIDATION is minimum for M = 23. Here, after M = 23, the graph begins to rise again indicating the increasing values of error if we increase the degree any further. Hence the optimal values the model parameters are:
M = 23, λ = 0.001 and ERMS = 0.483897


 COMPARISON WITH STANDARD PACKAGES
-----------------------------------
These results were compared with the standard packages available publically. The Neural Network package available with Matlab was chosen and Function Fitting and Pattern Recognition packages were used to compare with the linear regression model used above with varying results.

1. FUNCTION FITTING:
As the ERMS for this neural network is (MSETESTING)1/2, the preformance is evaluated on the testing samples which the network has not seen before, the error for comes out to be
ERMS = 0.528626

This package performs worse than the optimal linear regression model devised above as the ETESTING for the regression model is lesser than the error of this package.

2. PATTERN RECOGNITION NEURAL NETWORK:
A pattern recognition neural network of 30 nodes was used to learn from our original dataset of 46 features as showing in the figure. Learning was done on 70% as training, 10% as validation and rest 20% as testing, the performance as much better than the Function Fitting package above.

As the ERMS for this neural network is (MSETESTING)1/2, the preformance is evaluated on the testing samples which the network has not seen before, the error for comes out to be
ERMS = 0.307636

This package performs approximately 1.5 times better than the optimal linear regression model formulated above, as the ERMSTESTING is almost half of that found by the regression model.

