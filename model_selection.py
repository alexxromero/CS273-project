from __future__ import absolute_import, division, print_function, unicode_literals

import keras
import tensorflow as tf
import h5py
import numpy as np
from keras.utils import to_categorical
from keras.datasets import cifar10
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from AR_functions import linear_svm, polynomial_svm, define_model

#Load original cifar10 dataset
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

from sklearn.utils import shuffle

train_images_s, train_labels_s = shuffle(train_images,train_labels)

xval_samples = 5000
train_images_xval = train_images_s[:xval_samples] 
train_labels_xval = train_labels_s[:xval_samples] 

nsplits=5
kf = KFold(n_splits=nsplits, shuffle=True)

train_images_xval_svm = train_images_xval.reshape(train_images_xval.shape[0], 32*32*3) / 255

# change to whatever the user wants
c_values = [1e-3,1e-2, 1e-1, 1, 1e1, 1e2, 1e3,1e4,1e5]

acc_train_lin_array = np.empty(len(c_values))
acc_test_lin_array = np.empty(len(c_values))
acc_train_poly_array = np.empty(len(c_values))
acc_test_poly_array = np.empty(len(c_values))

for i,c in enumerate(c_values):

    acc_train_lin  = 0
    acc_test_lin   = 0
    acc_train_poly = 0
    acc_test_poly  = 0
    j = 1 
    for train, test in kf.split(train_images_xval_svm):
                
        print("Fold",j,"/",nsplits,"of regularization", i+1,"/",len(c_values))
        j=j+1
        x_train, y_train = train_images_xval_svm[train], train_labels_xval[train]
        
        x_test, y_test = train_images_xval_svm[test], train_labels_xval[test]
   
  
        train_scores, test_scores = linear_svm(c, x_train, y_train.ravel(), x_test, y_test.ravel())
        acc_train_lin = acc_train_lin + train_scores
        acc_test_lin = acc_test_lin + test_scores
        
       
        train_scores, test_scores = polynomial_svm(c, x_train, y_train.ravel(), x_test, y_test.ravel())
        acc_train_poly = acc_train_poly + train_scores
        acc_test_poly = acc_test_poly + test_scores

    acc_train_lin, acc_train_poly = acc_train_lin/nsplits, acc_train_poly/nsplits
    acc_test_lin, acc_test_poly = acc_test_lin/nsplits, acc_test_poly/nsplits


    acc_train_lin_array[i] = acc_train_lin
    acc_test_lin_array[i]  = acc_test_lin
    acc_train_poly_array[i] = acc_train_poly
    acc_test_poly_array[i] = acc_test_poly
    

plt.figure()
plt.plot(c_values, acc_train_lin_array,c='b', label="Train-LinearSVM")
plt.plot(c_values, acc_train_poly_array, c='c',label="Train-PolySVM")
plt.plot(c_values, acc_test_lin_array,c='r',label="Test-LinearSVM")
plt.plot(c_values, acc_test_poly_array, c='y',label="Test-PolySVM")
plt.title("SVM Accuracy vs C")
plt.legend()
plt.xlabel("C")
plt.ylabel("Accuracy")
plt.ylim(0,1.05)
plt.xscale("log")
plt.savefig('./SVM.eps')
#
#

train_images_xval_nn = train_images_xval / 255
train_labels_xval_nn = to_categorical(train_labels_xval, num_classes=10)
from keras.callbacks import EarlyStopping

epochs=100
learning_rate = [1e-5,5e-5,1e-4,5e-4,1e-3,5e-3,1e-2]

acc_train_log_array = np.empty(len(learning_rate))
acc_test_log_array = np.empty(len(learning_rate))
acc_train_relu_array = np.empty(len(learning_rate))
acc_test_relu_array = np.empty(len(learning_rate))

callbacks = [
         EarlyStopping(monitor='val_loss', patience=4)
]

for i, lr in enumerate(learning_rate):
 
 acc_train_relu = 0
 acc_test_relu =0 
 acc_train_log = 0
 acc_test_log = 0

 for train, test in kf.split(train_images_xval_nn):
        
        x_train, y_train = train_images_xval_nn[train], train_labels_xval_nn[train]
        
        x_test, y_test = train_images_xval_nn[test], train_labels_xval_nn[test]

        model_log = define_model("log",lr)

        history = model_log.fit(x_train, y_train, 
                    epochs=epochs,
                    batch_size=128,
                    validation_data=(x_test, y_test), callbacks=callbacks)

        train_scores = model_log.evaluate(x_train,y_train)
        test_scores = model_log.evaluate(x_test,y_test)

        acc_train_log = acc_train_log + train_scores[1]
        acc_test_log = acc_test_log + test_scores[1]

     
        model_log = define_model("relu", lr)

        history = model_log.fit(x_train, y_train, 
                    epochs=epochs,
                    batch_size=128,
                    validation_data=(x_test, y_test), callbacks=callbacks)

        train_scores = model_log.evaluate(x_train,y_train)
        test_scores = model_log.evaluate(x_test,y_test)

        acc_train_relu = acc_train_relu + train_scores[1]
        acc_test_relu = acc_test_relu + test_scores[1]

 acc_train_log_array[i]=acc_train_log/nsplits
 acc_test_log_array[i] = acc_test_log/nsplits
 acc_train_relu_array[i]=acc_train_relu/nsplits
 acc_test_relu_array[i] = acc_test_relu/nsplits

plt.figure()
plt.plot(learning_rate, acc_train_log_array,c='k', label="Train log")
plt.plot(learning_rate, acc_test_log_array, c='r',label="Test log")
plt.plot(learning_rate, acc_train_relu_array,c='b', label="Train relu")
plt.plot(learning_rate, acc_test_relu_array, c='g',label="Test relu")
plt.ylim(0,1.05)
plt.title("SimpleNet Accuracy vs Learning Rate")
plt.legend()
plt.xlabel("Learning Rate")
plt.ylabel("Accuracy")
plt.xscale("log")
plt.savefig('./simplenetXval.eps')

