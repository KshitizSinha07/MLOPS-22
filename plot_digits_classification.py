"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

###############################################################################
# Digits dataset
# --------------
#
# The digits dataset consists of 8x8
# pixel images of digits. The ``images`` attribute of the dataset stores
# 8x8 arrays of grayscale values for each image. We will use these arrays to
# visualize the first 4 images. The ``target`` attribute of the dataset stores
# the digit each image represents and this is included in the title of the 4
# plots below.
#
# Note: if we were working from image files (e.g., 'png' files), we would load
# them using :func:`matplotlib.pyplot.imread`.


# model hyperparams
GAMMA = 0.001
C = 0.5

#1. settin the ranges of hyperparameters


#2. train for every combination of hyper parameter values

#3. train the mode1
#4. compute the accuracy on validations set
#5. Identify the best combination of hyper parameters for which validation set acuracy is the highest
#6. Report the test set accuracyu with that best model


gamma_list=(0.001,0.01,0.002,0.005 )
c_list=(0.1,1,5,2,10,0.5)
h_params_list=[{'gamma':g,'c':C} for g in gamma_list for C in c_list ]

train_frac=0.8
test_frac=0.1
dev_frac=0.1



digits = datasets.load_digits()

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)

# Classification

# flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

#PART: define train/dev/test splits of experiment protocol
# train to train model
# dev to set hyperparameters of the model
# test to evaluate the performance of the model
dev_test_frac = 1-train_frac
X_train, X_dev_test, y_train, y_dev_test = train_test_split(
    data, digits.target, test_size=dev_test_frac, shuffle=True
)
X_test, X_dev, y_test, y_dev = train_test_split(
    X_dev_test, y_dev_test, test_size=(dev_frac)/dev_test_frac, shuffle=True
)

#PART: Define the model
# Create a classifier: a support vector classifier
clf = svm.SVC()



#PART: setting up hyperparameter
#hyper_params = {'gamma':GAMMA, 'c':C}
best_acc=-1
best_model=None
best_hyperparams=None

for hyper_params in h_params_list:
    #print(hyper_params)   

    clf.set_params(**hyper_params)

    # Learn the digits on the train subset
    clf.fit(X_train, y_train)

    # Predict the value of the digit on the test subset
    predicted_dev = clf.predict(X_dev)

    current_acc = metrics.accuracy_score(y_pred=predicted_dev, y_true=y_dev)

    if current_acc> best_acc:
        best_acc=current_acc
        best_model=clf
        best_hyperparams=hyper_params

print(best_acc)
print(best_hyperparams)


predicted_dev = clf.predict(X_test)
current_acc = metrics.accuracy_score(y_pred=predicted_dev, y_true=y_test)


    ###############################################################################
    # Below we visualize the first 4 test samples and show their predicted
    # digit value in the title.
"""
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, prediction in zip(axes, X_test, predicted):
        ax.set_axis_off()
        image = image.reshape(8, 8)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"Prediction: {prediction}")
"""
    ###############################################################################
    # :func:`~sklearn.metrics.classification_report` builds a text report showing
    # the main classification metrics.

print(
        f"Classification report for classifier {clf}:\n"
        f"{metrics.classification_report(y_test, predicted)}\n"
    )

###############################################################################
# We can also plot a :ref:`confusion matrix <confusion_matrix>` of the
# true digit values and the predicted digit values.

#disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
#disp.figure_.suptitle("Confusion Matrix")
#print(f"Confusion matrix:\n{disp.confusion_matrix}")

#plt.show()
