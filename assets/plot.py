from lr import LogisticRegressionClassifier
import numpy as np
import matplotlib.pyplot as plt

num_epoch = 200
epochs = np.arange(1,num_epoch+1)

model1_train_file = 'eric_largeoutput/model1_formatted_train.tsv'
model1_valid_file = 'eric_largeoutput/model1_formatted_valid.tsv'

model2_train_file = 'eric_largeoutput/model2_formatted_train.tsv'
model2_valid_file = 'eric_largeoutput/model2_formatted_valid.tsv'


lr_classifier = LogisticRegressionClassifier()
train_X,train_Y = lr_classifier.parse_data(model2_train_file)
valid_X,valid_Y = lr_classifier.parse_data(model2_valid_file)

_,all_theta = lr_classifier.stochastic_gradient_descent(train_X,train_Y,num_epoch)

train_nll = []
valid_nll = []

for theta in all_theta:
    train_nll.append(lr_classifier.negative_loglikelihood_function(theta,train_X,train_Y))
    valid_nll.append(lr_classifier.negative_loglikelihood_function(theta,valid_X,valid_Y))

# train_X,train_Y = lr_classifier.parse_data(model1_train_file)
# valid_X,valid_Y = lr_classifier.parse_data(model1_valid_file)

# train_nll = []
# valid_nll = []

# for theta in all_theta:
#     train_nll.append(lr_classifier.negative_loglikelihood_function(theta,train_X,train_Y))
#     valid_nll.append(lr_classifier.negative_loglikelihood_function(theta,valid_X,valid_Y))

plt.plot(epochs,train_nll, ls='--', marker='o', label='Train Negative Log Likelihood')
plt.plot(epochs,valid_nll,  ls='-', marker='v', label='Valid Negative Log Likelihood')
plt.xlabel('Epoch')
plt.ylabel('Negative Log Likelihood')
plt.title('Large Dataset (Model 2) Train/Valid Negative Log Likelihood')
plt.legend()
plt.show()
