#(C) Copyright Syd Logan 2021
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#
#You may obtain a copy of the License at
#http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import matplotlib.pyplot as plt
import numpy as np
import math

class Data():
    '''
    class to hold data and perform the operations needed by this example
    '''
    def __init__(self):
        self._x = None
        self._y = None
        self._data = None
        self._u = None
        self._d = None
        self._v = None
        self._data2d = None
        self._data_m = None

    def loadX(self, filename):
        '''
        in octave:
        x = load('ex2x.dat');
        '''
        t = []
        f = open(filename) 
        for v in f.read().split('\n'):
            try:
                t.append(float(v)) 
            except:
                pass
            
        self._x = np.array(t)
        f.close()

    def loadY(self, filename):
        '''
        in octave:
        y = load('ex2y.dat');
        '''

        t = []
        f = open(filename) 
        for v in f.read().split('\n'):
            try:
                t.append(float(v)) 
            except:
                pass
            
        self._y = np.array(t)
        f.close()

    def getX(self):   
        return self._x

    def getY(self):
        return self._y

    def length(self):
        '''
        in octave:
        m = length(y); % store the number of training examples
        '''
        return self._y.shape[0]   # tuple (m, n) - return m

    def normalize(self):
        '''
        in octave:
        data=[x,y];
        mu = mean(data); % each row is a data sample
        data_m = (data - repmat(mu, m, 1));
        '''     
    
        self._data = np.vstack((self._x, self._y)).T
        self._mu = np.array([np.mean(self._x), np.mean(self._y)])
        tile = np.tile(self._mu, (self.length(), 1))
        self._data_m = self._data - tile
        

    def computeCovariance(self):
        '''
        in octave:
        Sx = cov(data_m, 1); % 0 = normalize with N - 1, 1 = normalize with N
        '''
        self._covariance = np.cov(self._data_m, rowvar=False, bias=True)

    def getCovariance(self):
        return self._covariance
    
    def computeSVDEcon(self):
        '''
        in octave:
        [U, D, V] = svd(Sx, 'econ');
        '''
        self._u, self._d, self._v = np.linalg.svd(self._covariance, full_matrices = False)

    def getSVD(self):
        return (self._u, self._d, self._v)

    def computeDecorrelated2D(self):
        '''
        in octave:
        data_2d=data_m*V;
        '''

        self._data_2d = self._data_m.dot(self._v)

    def getDecorrelated2D(self):
        return self._data_2d

    def getEigenData(self):
        '''
        in octave:
        largest_eigenvec = V(:,1);
        largest_eigenval = D(1,1);
        medium_eigenvec = V(:,2);
        medium_eigenval = D(2,2);

        remember in Python, indexes are 0 based. And numpy arrays are indexed
        with [], not ()
        '''
        self._largestEigenValue = self._d[0]
        self._largestEigenVec = self._v[:,0]
        return (self._v[:,0], self._d[0], self._v[:,1], self._d[1])

    def getLargestEigenValue(self):
        return self._largestEigenValue

    def getLargestEigenVector(self):
        return self._largestEigenVec

    def compute1DData(self):
        '''
        in octave:
        eigenvec_1d=largest_eigenvec;
        data_1d = data_m*eigenvec_1d;
        '''
        self._eigenvec_1d = self._largestEigenVec
        self._data1d = self._data_m * self._eigenvec_1d

    def get1DData(self):
        return self._data1d

    def getMu(self):
        '''
        in octave:
        X0=mu(1);
        Y0=mu(2);
        '''
        return self._mu

class Outputs():
    '''
    class containing various output functions, generally one per
    figure created by the example
    '''

    def plotTrainingData(self, fig, data):
        '''
        in octave:
        figure % open a new figure window
        %plot your training set (and label the axes):
        plot(x, y, 'o');
        ylabel('Height in meters')
        xlabel('Age in years')
        title('Original 2D data');
        hold on
        '''
        fig.plot(data.getX(), data.getY(), 'o')
        fig.setYLabel('Height in meters')
        fig.setXLabel('Age in years')
        fig.setTitle('Original 2D data')

    def plotDecorrelated2D(self, figure, data_2d):
        '''
        in octave:
        plot(data_2d(:,1), data_2d(:,2), 'o');
        colormap(gray);
        hold on;
        '''

        figure.plot(data_2d[:,0], data_2d[:, 1], 'o')

    def plotEigenVectors2D(self, figure, D):
        '''
        in octave:
        % Plot the eigenvectors (which are now the axes
        h=quiver( 0,0,1*sqrt(D(1,1)), 0*sqrt(D(1,1)), '-m', 'LineWidth',3);
        quiver(0, 0, 0*sqrt(D(2,2)), 1*sqrt(D(2,2)), '-g', 'LineWidth',3);
        hold on;
        '''
        figure.quiver(0, 0, 1*math.sqrt(D[0,0]), 0*math.sqrt(D[0,0]), 'm')
        # XXX to be compat with octave output, appears we need to scale to 
        # be a unit vector here before calling quiver.
        D = D / np.sqrt(np.sum(D**2))
        figure.quiver(0, 0, 0*math.sqrt(D[1,1]), 1*math.sqrt(D[1,1]), 'g')
    

    def showEigenQuivers(self, figure, data):
        '''
        in octave:
        quiver(X0,Y0,  largest_eigenvec(1)*sqrt(largest_eigenval), largest_eigenvec(2)*sqrt(largest_eigenval), '-m', 'LineWidth',3);
        quiver(X0, Y0,  medium_eigenvec(1)*sqrt(medium_eigenval), medium_eigenvec(2)*sqrt(medium_eigenval), '-g', 'LineWidth',3);
        '''
        mu = data.getMu()
        X0 = mu[0]
        Y0 = mu[1]
        eigenData = data.getEigenData()
        largest_eigenvec = eigenData[0]
        largest_eigenval = eigenData[1]
        medium_eigenvec = eigenData[2]
        medium_eigenval = eigenData[3]
        figure.quiver(X0,  Y0, largest_eigenvec[0]*math.sqrt(largest_eigenval),
largest_eigenvec[1]*math.sqrt(largest_eigenval), 'm');
        figure.quiver(X0, Y0,  medium_eigenvec[0]*math.sqrt(medium_eigenval),
medium_eigenvec[1]*math.sqrt(medium_eigenval), 'g');

    def plotProjection(self, figure, data1D, largestEigen):
        '''
        in octave:

        % Plot the 1D data
        figure;
        plot(data_1d, repmat(0, size(data_1d',1), 1) , 'o');
        colormap(gray);

        % Plot the eigenvector
        hold on;
        quiver(0, 0, 1*sqrt(largest_eigenval), 0*sqrt(largest_eigenval), '-m', 'LineWidth',3);
        hold on;

        % Set the axis labels
        hXLabel = xlabel('x');
        hYLabel = ylabel('y');

        title('Projected 1D data');
        grid on;
        '''
        shape = data1D.shape
        tile = np.tile(0, (shape[0], 1))
        figure.plot(data1D, tile, 'o');
        figure.quiver(0,  0, 1*math.sqrt(largestEigen), 0*math.sqrt(largestEigen), 'm')
        figure.setXLabel('x')
        figure.setYLabel('y')
        figure.setTitle('Projected 1D data')

    def dumpResults(self, data):
        '''
        in octave:
        disp("Matrix of Covariance"), disp(Sx);
        disp("Major Principal Component"),disp(V(:,1));
        disp("End of the program");
        '''
        print("Matrix of Covariance\n{}".format(data.getCovariance()))
        print("Major Principal Component\n{}".format(data.getLargestEigenVector()))
        print("End of the program");

class Figure():
    ''' 
    class that abstracts an octave figure and some of its functionality,
    such as creation, display, plots and quivers
    '''

    def __init__(self, serial):
        self._figure = plt.figure(serial)
        self._serial = serial 

    def hold(self):
        '''
        in octave:
        hold
        '''
        plt.show()

    def quiver(self, x, y, u, v, color):
        '''
        in octave:
        quiver(X0,Y0,  largest_eigenvec(1)*sqrt(largest_eigenval), largest_eigenvec(2)*sqrt(largest_eigenval), '-m', 'LineWidth',3);
        '''
        plt.quiver(x, y, u, v, angles="xy", scale_units='xy', headwidth=10, scale=1, color=color)

    def setXLabel(self, val):
        '''
        in octave:
        xlabel('Age in years')
        '''
        plt.xlabel(val)

    def setYLabel(self, val):
        '''
        in octave:
        ylabel('Height in meters')
        '''
        plt.ylabel(val)

    def setTitle(self, val):
        '''
        in octave:
        title('Original 2D data');
        '''
        plt.title(val)

    def grid(self, val):  # val is True: show grid, False: hide grid
        ''' 
        in octave:
        grid
        
        '''
        plt.grid(val)

    def plot(self, x, y, attrib):
        '''
        in octave:
        plot(x, y, 'o');
        '''
        plt.plot(x, y, attrib)

def main():
    data = Data()
    data.loadX('ex2x.dat');
    data.loadY('ex2y.dat');
    outputs = Outputs()

    # Figure 1, training data and its eigenvectors

    fig = Figure(1)
    outputs.plotTrainingData(fig, data)
    data.normalize()
    data.computeCovariance()
    data.computeSVDEcon()
    outputs.showEigenQuivers(fig, data)

    # Figure 2, decorrelated 2D

    fig = Figure(2)
    data.computeDecorrelated2D()
    data2D = data.getDecorrelated2D()
    outputs.plotDecorrelated2D(fig, data2D)
    outputs.plotEigenVectors2D(fig, data2D)

    # Figure 3, PCA projection to 1D

    fig = Figure(3)
    data.compute1DData()
    data1D = data.get1DData()
    largestEigenVal = data.getLargestEigenValue()
    outputs.plotProjection(fig, data1D, largestEigenVal)

    outputs.dumpResults(data)
    fig.hold()

if __name__ == "__main__":
    main()

'''
for references, the original octave sources that inspired this port:

x = load('ex2x.dat');
y = load('ex2y.dat');
m = length(y); % store the number of training examples
figure % open a new figure window
%plot your training set (and label the axes):
plot(x, y, 'o');
ylabel('Height in meters')
xlabel('Age in years')
title('Original 2D data');
hold on

data=[x,y];
mu = mean(data); % each row is a data sample
data_m = (data - repmat(mu, m, 1));
% ====================== YOUR CODE HERE ======================
%Compute the covariance matrix- one line of code
% Note: When computing the covariance matrix, remember to divide by m (the
% number of examples).
% Sx = 

Sx = cov(data_m, 1); % 0 = normalize with N - 1, 1 = normalize with N

 %  Then use the "svd" function to compute the eigenvectors
%     and eigenvalues of the covariance matrix. 
%              [U,D,V]  =

[U, D, V] = svd(Sx, 'econ');

% ====================== YOUR CODE END ======================


largest_eigenvec = V(:,1);
largest_eigenval = D(1,1);
medium_eigenvec = V(:,2);
medium_eigenval = D(2,2);
X0=mu(1);
Y0=mu(2);
quiver(X0,Y0,  largest_eigenvec(1)*sqrt(largest_eigenval), largest_eigenvec(2)*sqrt(largest_eigenval), '-m', 'LineWidth',3);
quiver(X0, Y0,  medium_eigenvec(1)*sqrt(medium_eigenval), medium_eigenvec(2)*sqrt(medium_eigenval), '-g', 'LineWidth',3);
data_2d=data_m*V;
% Plot the 2D decorrelated data 
figure;
 plot(data_2d(:,1), data_2d(:,2), 'o');
colormap(gray);
hold on;
% Plot the eigenvectors (which are now the axes
h=quiver( 0,0,1*sqrt(D(1,1)), 0*sqrt(D(1,1)), '-m', 'LineWidth',3);
quiver(0, 0, 0*sqrt(D(2,2)), 1*sqrt(D(2,2)), '-g', 'LineWidth',3);
hold on;

%%%%%%%%%%%%% PROJECT THE DATA ONTO THE LARGEST EIGENVECTOR %%%%%%%%%%%

eigenvec_1d=largest_eigenvec;

data_1d = data_m*eigenvec_1d;

% Plot the 1D data
figure;
 plot(data_1d, repmat(0, size(data_1d',1), 1) , 'o');
colormap(gray);

% Plot the eigenvector
hold on;
quiver(0, 0, 1*sqrt(largest_eigenval), 0*sqrt(largest_eigenval), '-m', 'LineWidth',3);
hold on;

% Set the axis labels
hXLabel = xlabel('x');
hYLabel = ylabel('y');

title('Projected 1D data');
grid on;
disp("Matrix of Covariance"), disp(Sx);
disp("Major Principal Component"),disp(V(:,1));
disp("End of the program");
'''
