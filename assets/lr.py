import argparse
import numpy as np
import copy

class LogisticRegressionClassifier():
    def __init__(self):
        self.theta = {0:0.0} # bias
        self.X = []
        self.Y = []

    def fit(self,train_input):
        self.X, self.Y = self.parse_data(train_input)

    def train(self,num_epoch):
        self.theta = self.stochastic_gradient_descent(self.X,self.Y,num_epoch)

    def sigmoid(self,z):
        return 1.0 / (1.0+np.exp(-z))

    def sparse_dot(self,theta,x):
        lm = 0.0
        for idx,value in x.items():
            if idx in theta:
                lm += value*theta[idx]
            else:
                continue
        return lm
    
    def update_theta(self,theta,gradient,alpha):
        for idx,value in gradient.items():
            if idx in theta:
                theta[idx] -= alpha*gradient[idx]
            else:
                theta[idx] = -alpha*gradient[idx]
        return theta
    
    def gradient_funciton(self,theta,y,x,N):
        lm = self.sparse_dot(theta,x)
        gradient = {}
        for key in x:
            gradient[key] = (-y + self.sigmoid(lm))/N
        return gradient

    def stochastic_gradient_descent(self,X,Y,num_epoch):
        alpha = 0.1
        theta = {0:0.0}
        count = 0
        N = len(Y)
        while count < num_epoch:
            count += 1
            for i in range(len(Y)):
                y = Y[i]
                x = X[i]
                gradient = self.gradient_funciton(theta,y,x,N)
                theta = self.update_theta(theta,gradient,alpha)
        return theta

    def transform(self,input_file,output_file):
        with open(output_file,mode='w') as f:
            X,Y = self.parse_data(input_file)
            for i in range(len(Y)):
                x = X[i]
                lm = self.sparse_dot(self.theta,x)

                if lm > 0:
                    f.write('1\n')
                else:
                    f.write('0\n')

    def parse_data(self,train_input):
        X = []
        Y = []
        with open(train_input,mode='r') as f:
            for line in f.readlines():
                splitted_line = line.strip().split('\t')
                Y.append(float(splitted_line[0]))

                reformed_x = {0:1.0} # bias
                x = splitted_line[1:]
                for i in range(len(x)):
                    x[i] = x[i].split(':')
                    key = x[i][0]
                    value = x[i][1]
                    reformed_x[int(key)+1] = float(value)
                X.append(reformed_x)
        return X,Y
    
    def negative_loglikelihood_function(self,theta,X,Y):
        N = len(Y)
        nll = 0
        for i in range(N):
            x = X[i]
            y = Y[i]
            lm = self.sparse_dot(theta,x)
            nll += -y*lm + np.log(1+np.exp(lm))
        return nll/N
        
class Evaluator():
    def __init__(self):
        self.err = None
    
    def evaluate(self,origin_file,prediction_file):
        origin = []
        prediction = []

        with open(origin_file,mode='r') as f:
            for line in f.readlines():
                label = line.strip().split('\t',maxsplit=1)[0]
                origin.append(label)

        with open(prediction_file,mode='r') as f:
            for line in f.readlines():
                label = line.strip()
                prediction.append(label)
        
        total = len(origin)
        wrong = 0
        for i in range(total):
            if origin[i] != prediction[i]:
                wrong += 1
        
        self.err = wrong/total
        return self.err

def parse_args():
    """
    Parse input positional arguments from command line
    :return: args - parsed arguments
    """
    parser = argparse.ArgumentParser('Sentiment Polarity Analyzer - Binary Logistic Regression')
    parser.add_argument('formatted_train_input',help='path to the formatted training input .tsv file')
    parser.add_argument('formatted_validation_input',help='path to the formatted validation input .tsv file')
    parser.add_argument('formatted_test_input',help='path to the formatted test input .tsv file')
    parser.add_argument('dict_input',help='path to the dictionary input .txt file')
    parser.add_argument('train_out',help='path to output .labels file to which the prediction on the training data should be written')
    parser.add_argument('test_out',help='path to output .labels file to which the prediction on the test data should be written')
    parser.add_argument('metrics_out',help='path of the output .txt file to which metrics such as train and test error should be written')
    parser.add_argument('num_epoch',help='integer specifying the number of times SGD loops through all of the training data')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    lr_classifier = LogisticRegressionClassifier()
    print('Fitting data, please wait...')
    lr_classifier.fit(args.formatted_train_input)
    print('Fitting data completed!\n')

    print('Training data, please wait...')
    lr_classifier.train(num_epoch=int(args.num_epoch))
    print('Training Model Completed!\n')


    lr_classifier.transform(input_file=args.formatted_train_input,output_file=args.train_out)
    lr_classifier.transform(input_file=args.formatted_test_input,output_file=args.test_out)

    evaluator = Evaluator()
    train_err = evaluator.evaluate(origin_file=args.formatted_train_input,prediction_file=args.train_out)
    test_err = evaluator.evaluate(origin_file=args.formatted_test_input,prediction_file=args.test_out)

    with open(args.metrics_out,mode='w') as f:
        f.write('error(train): ' + str(train_err) + '\n')
        f.write('error(test): ' + str(test_err) + '\n')