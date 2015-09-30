__author__ = 'graham'
# Graham Traines
# 1 March 2015

import numpy as np
import scipy as sp
import matplotlib.pyplot as mpl


class DataSet:
    def __init__(self):
        self.classification = []
        self.color = []
        self.X = []
        return

    def generate_from_text(self, file_location, delimiter):
        data = sp.genfromtxt(file_location, delimiter=delimiter)

        #print data.shape

        # Need to get everything after the headers
        # Alternatively we could slice a very small data set out of the larger data set for debugging
        # we need data point from the red and blue groups so the slices are disjoint

        # I could do this in one step but I don't
        x1 = data[1:, 0]
        x2 = data[1:, 1]
        cluster = data[1:, 2]
        self.classification = [1 if point == 1 else -1 for point in cluster]

        ones = [1 for x in x1]
        self.X = np.array(zip(ones, x1, x2))

        self.color = ["red" if point == 1 else "blue" for point in self.classification]

        return

class Perceptron:
    def __init__(self, dataset):
        #init
        self.dataset = dataset
        # M = size of dataset
        self.M = len(self.dataset.X)
        self.X = []
        self.g_in_h = np.zeros(3)
        self.iteration = 0
        self.epsilon = 1

        return
    def learn(self, epsilon):

        #epsilon is the learning increment
        self.epsilon = epsilon

        # initialize weights to 0
        hypothesis = np.zeros(3)

        current_error = 1.0
        while current_error != 0:
            self.iteration += 1

            x, classification_err = self.get_random_misclassified_point(hypothesis)
            #Update the weights
            delta = self.epsilon * classification_err
            #print delta*x
            hypothesis += delta*x
            #print hypothesis
            current_error = self.err_in_sample(hypothesis)
            #print "Current error: %f" % current_error

        self.g_in_h = hypothesis
        print "Iterations"
        print self.iteration
        return

    def get_random_misclassified_point(self, hypothesis):
        # return the first misclassified point we come across
        # trying to speed up the selection process rather than cycling through the entire dataset every time
        # suggested on pg. 8 of Learning from Data
        for i in range(0, (len(self.dataset.X) - 1)):

            hypothesis_out = int(np.sign(hypothesis.transpose().dot(self.dataset.X[i])))
            # print "Hypothesis guess:"
            # print hypothesis_out
            # print "Classification: "
            # print self.dataset.classification[i]
            if hypothesis_out != self.dataset.classification[i]:
                #per Tom Mitchell's book, delta of wi should = learning rate * (target output - perceptron output) * xi
                return (self.dataset.X[i], (self.dataset.classification[i] - hypothesis_out))

    def err_in_sample(self, hypothesis):
        # return % misclassified points

        n_misclassified_points = 0
        for i in range(0, (len(self.dataset.X) - 1)):
            if int(np.sign(
                    hypothesis.transpose().dot(
                        self.dataset.X[i]
                    )
            )) != self.dataset.classification[i]:
                n_misclassified_points += 1
        e_in = n_misclassified_points / float(self.M)
        return e_in

class Plotter:
    def __init__(self, perceptron):
        self.p = perceptron
        return

    def plot_data_points(self, plot_title, x_axis_label, y_axis_label):

        self.plot_hypothesis_line()
        mpl.scatter(self.p.dataset.X[:, 1],
                    self.p.dataset.X[:, 2],
                    c=self.p.dataset.color)
        mpl.title((plot_title + (" N = %s, Iteration = %s" % (str(self.p.M), str(self.p.iteration)))))
        mpl.xlabel(x_axis_label)
        mpl.ylabel(y_axis_label)
        mpl.autoscale(tight=True)
        mpl.grid()
        mpl.show()
        return

    def plot_hypothesis_line(self):

        g_in_h = self.p.g_in_h

        m, b = -g_in_h[1]/g_in_h[2], -g_in_h[0]/g_in_h[2]
        x = np.linspace(-1, 1)
        mpl.plot(x, m*x+b, 'k-')

        return

ds = DataSet()
ds.generate_from_text("/home/graham/Desktop/MachineLearning/Mod3/CSCI_7090_PerceptronData_N2000_IT2683.csv", ",")

perceptron = Perceptron(ds)
perceptron.learn(.25)

plot = Plotter(perceptron)
plot.plot_data_points("Graham Traines", "X1", "X2")
