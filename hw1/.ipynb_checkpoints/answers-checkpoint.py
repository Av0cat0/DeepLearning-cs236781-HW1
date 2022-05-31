r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""


1. The answer is **False**.
in-sample errors are defined as the error rate we get on the **train** set we used to build the predictor while the out-sample errors are the errors on the data set (such as the validation set) that wasn’t used to build the predictor. Because the subset which was used to build the predictor is the train set and not the test set, the test set is not able to tell us the estimatiom for the in sample errors.

2. The answer is **False**.
There are many reasons for differs between different train_test_split executions such as:

    a. *non-random split*, if the split is known and not random, I can decide how the subsets will be created resulting in  controlling the in-sample error (not only the value but also the learning process) so this non-random train-test split is different and less useful than a regular (randomed) train_test_split.
    
    b. *one of the subsets contains specific data in regards to other subsets*, If a random (or non-random) train-test split generates two subsets with a feature that has a limited boundaries. for example, if the dataset includes a list of people and their income/genders and after the splitting, the train set has only people with a specific income or the test set contains only women while in the train set we has the same amount (the amount is not relevant, it's just an example) then we might result in overfitting therefore making train-test splits not equally useful as other train-test splits.
    
3. The answer is **True**
Our CV algorithm generates its own test-sets to wotk with (which is the subset that was not built out of k-1 subsamples dedicated for training), if we used the same test-set we ignore the notion of learning the hyperparameters and just determine it, which is a bad idea because we ignore by that a learning technique. 
4.The answer is **False**
After performing cross-validation, we will estimate our generalization performance by disjoint subsets that has not affected our model. In cross-validation, for each k we'll choose, we'll get that the validation set in the first iteration must be a part of the train set from the second iteration (Can be proved in induction for every indexed i validation set) so since the data completely processed during the training, the validation set can't represent  a proxy for the model's generalization error.
    

"""

part1_q2 = r"""

His approach is **not** justified because he fits the hyperparameter $\lambda$ by himself so he could still have overfitting in his model and the testing results won't be concluded from a general dataset-splitting, but only from a limited splitting of datasets therefore the result won't apply for the generalization problem.


"""

# ==============
# Part 2 answers

part2_q1 = r"""

Let's investigate the value of *k* according to our `K-fold` algorithm:

The higher the value of k, the more neighbors will take account when calculating the label of a specific value (dataset).
When we take a small value for k, for example k=1:
The model assign to each value its closest neighbor, resulting in overfitting and a model which is sensitive to errors and noises (a small noise can change the label of the value to a different label) therefore we won't get optimal result and a huge generalization gap.

On the other hand, if the value of k is extremely high, we'll calculate all the k neighbors of the value and that means we'll eventually take the "bad" labels (all the labels which are not the true label of our value) so we risk in underfitting and our predictor becomes just a mean function, labeling all the new/test samples as the one that appears the most in the dataset $\Rightarrow$ resulting in bad model.

So increasing k up to the estimated "sweet spot" will improve generalization for unseen data as it will increase the generalization capacity of the model but increasing further than that will result in bad model.



"""

part2_q2 = r"""


a. The detailed process is problematic in general because we can pick a model that fits perfectly with the train set but with extremely overfitting that won't fit so well for new datasets (low generalization capacity) while the K-fold CV searches for best train-set to avoid generalization gap

b. In this process, we determine what is the train set and what is the test set. Instead of looking for the best split for train-test sets, this process looks for the best model given train-test sets. This also can result in generalization gap since there could be important/massive difference between the train and the test sets



"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""

Changing the value of $\Delta$ re-scales the weights, resulting in changing our model to a different one.
The purpose of $\Delta$ is to make sure there's distance between the sample's prediction to other possible perdictions in our class,
So if we'll change $\Delta$ we'll need to change our *$W$* to give different weights for different samples so they'll still have proper margin from the boundary the model created and we could sperate them with a distance from other classes, therefore $\Delta$'s value is arbitrary.

"""

part3_q2 = r"""

1. The model is trying to learn the right linear hyperplane that will make the samples linear separated between different predictions in our class. in the visualization section's code, we can see that the class is the digits ${0,\dots,9}$ and some digits look alike so some of $W$ columns are similar by their values and that's making some bad predictions in our model and that's how the classification errors behave (Similarity between an image of a digit and some wrong prediction $\Rightarrow$ wrongly choose the wrong prediction according the $W$ $\Rightarrow$ making a classification error).

2. This interpretation is similar to kNN by the assumptions it makes: while the linear SVM classify samples by taking the argmax value of the provided scores (scores are generated by linear separation first and then measure the distance of the sample with the weight matrix and the hyperplane) the kNN model classify samples by the K most similar images. the difference is that linear SVM uses all its data to classify by the best score while kNN saves all the data (in training stage) but uses only the neighbors of the sample to classify.

"""

part3_q3 = r"""
1. The learning rate we chose for our training set is *good*, We can see some foundations for that:

    a. The loss is decreased monotonically without any peaks/hills
    
    b. The loss is converged to a smaller value than the validation loss (and the value 1) after less than 5 epochs, meaning each epoch we run afterwards will have smaller loss and it will only improve.
    
    We can also explain our answer by looking at what the graph will be look like if our learning rate wasn't good:

    If our learning rate was *too high*, it can cause undesirable divergent behavior to the loss function, making it miss and "jumping" over the minimum thus the loss function won't converge.
If our learning rate was *too low*, the training will be slow resulting in slower convergence in the loss function, that's because the weight matrix would be updated by tiny changes.

2. The model is *slightly overfitted* to the training set, we can conclude that from investigating the loss function: we can see that after only a few iterations the test loss  > training loss, meaning we're training our model in such complexity that the model focusing on decreasing the train samples loss, resulting in a better loss for the training sets instead of the test set. we can also conclude that from the accuracy graph: the train set accuracy is high while the test set is not that good, that means the model focused a bit too much on training instead of validating. the model is not *highly overfitting* because if it was, the accuracy of the test set was much lower because of training the model for the specific training sample.

"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""

The ideal pattern to see on a residual plot would be points scattered horizontally to all over the graph,
with the desire to arrive to $y = 0$ alike function.
that means thats our learning-predicton model predicts with a measured low-error for our given data.
$min(|y-y\hat|) = 0$
Based on the residual plot we can say that the fitness of last trained model is much better then the
"five features - model" where you can see the data is scattered more around the fitted model.
moreover, we can base our answer solely on mse scoring as we got lower mse in the last model,
the overall result would imply whos the better model low-residual wise.

"""

part4_q2 = r"""

The effect of adding non-linear features would boost our data with more information
and therfore we'll get a better results.
1.The linear in linear regression refers to linear prediction model $Wb_i+b_0$ and not to the type of data (linear/non-linear) we use to 
fit this model.
2.Yes, as we know from previous courses : Every non linear model can be locally linearized by using Taylor expansion.
3.The descision boundary for adding non-linear features would still be an hyper-plane simply because this is how we train our model to be,
for example:
    $...W_3*b_3+W_2*b_2+W_1*b_1+b_0 = y$
this model right here is hyper-plane over the given features ($b_1-b_3$) linear or non-linear.

"""

part4_q3 = r"""

1. First the difference between the two is that np.logspace returns numbers spaced evenly on a log scale and np.linspace returns numbers evenly spaced over a specified interval. The meaning is that we can use np.logspace and get the same length of lambda and same number of iterations with much more variate values given np.linspace.
which means better training for our model with same effort in time and memory
    
2.  The model was fitted in total 180 time.
20 times for each lambda value,3 times for each degree and 3 times for each k-fold.
so we get $20*3*3 = 180$
"""

# ==============
