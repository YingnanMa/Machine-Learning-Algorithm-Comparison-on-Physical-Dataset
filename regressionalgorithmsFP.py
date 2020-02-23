from __future__ import division  # floating point division
import numpy as np
import math
import time

import utilitiesFP as utils

def l2err_squared(prediction,ytest):
    """ l2 error squared """
    return np.square(np.linalg.norm(np.subtract(prediction,ytest)))

def l2err(prediction,ytest):
    """ l2 error (i.e., root-mean-squared-error) """
    return np.linalg.norm(np.subtract(prediction,ytest))

# similar to l2err
def Frobenius_norm(X):
    return np.linalg.norm(X)

class Regressor:
    """
    Generic regression interface; returns random regressor
    Random regressor randomly selects w from a Gaussian distribution
    """

    def __init__( self, parameters={} ):
        """ Params can contain any useful parameters for the algorithm """
        self.params = {}
        self.reset(parameters)

    def reset(self, parameters):
        """ Reset learner """
        self.weights = None
        self.resetparams(parameters)

    def resetparams(self, parameters):
        """ Can pass parameters to reset with new parameters """
        self.weights = None
        try:
            utils.update_dictionary_items(self.params,parameters)
        except AttributeError:
            # Variable self.params does not exist, so not updated
            # Create an empty set of params for future reference
            self.params = {}

    def getparams(self):
        return self.params

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        self.weights = np.random.rand(Xtrain.shape[1])

    def predict(self, Xtest):
        """ Most regressors return a dot product for the prediction """
        ytest = np.dot(Xtest, self.weights)
        return ytest

class RangePredictor(Regressor):
    """
    Random predictor randomly selects value between max and min in training set.
    """

    def __init__( self, parameters={} ):
        """ Params can contain any useful parameters for the algorithm """
        self.params = {}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.min = 0
        self.max = 1
        
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        self.min = np.amin(ytrain)
        self.max = np.amax(ytrain)

    def predict(self, Xtest):
        ytest = np.random.rand(Xtest.shape[0])*(self.max-self.min) + self.min
        return ytest

class MeanPredictor(Regressor):
    """
    Returns the average target value observed; a reasonable baseline
    """
    def __init__( self, parameters={} ):
        self.params = {}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.mean = None
        
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        self.mean = np.mean(ytrain)

    def predict(self, Xtest):
        return np.ones((Xtest.shape[0],))*self.mean


class FSLinearRegression(Regressor):
    """
    Linear Regression with feature selection, and ridge regularization
    """
    def __init__( self, parameters={} ):
        self.params = {'features': [1,2,3,4,5]}
        self.reset(parameters)

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        # Dividing by numsamples before adding ridge regularization
        # to make the regularization parameter not dependent on numsamples
        numsamples = Xtrain.shape[0]
        
        # 2(a add a lambda*I for regularization 
        lambda_ = 0.00001
        Xtrain[:len(Xtrain[0])] = np.add( Xtrain[:len(Xtrain[0])], lambda_*np.identity(len(Xtrain[0])) )
        
        Xless = Xtrain[:,self.params['features']]
        self.weights = np.dot(np.dot(np.linalg.inv(np.dot(Xless.T,Xless)/numsamples), Xless.T),ytrain)/numsamples

    def predict(self, Xtest):
        Xless = Xtest[:,self.params['features']]
        ytest = np.dot(Xless, self.weights)
        return ytest

class RidgeLinearRegression(Regressor):
    """
    Linear Regression with ridge regularization (l2 regularization)
    TODO: currently not implemented, you must implement this method
    Stub is here to make this more clear
    Below you will also need to implement other classes for the other algorithms
    """
    def __init__( self, parameters={} ):
        # Default parameters, any of which can be overwritten by values passed to params
        self.params = {'regwgt': 0.01}
        self.reset(parameters)
        
    def learn(self, Xtrain, ytrain):
        dotX = np.dot(Xtrain.T,Xtrain)
        # regularize data
        dotX[:len(Xtrain[0])] = np.add( dotX[:len(Xtrain[0])], self.params['regwgt']*np.identity(len(Xtrain[0])) )
        # calculate weights
        self.weights = np.dot(np.dot(np.linalg.inv(dotX), Xtrain.T),ytrain)
        
    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        return ytest    

    
class LassoLinearRegression(Regressor):
    def __init__( self, parameters={} ):
        # Default parameters, any of which can be overwritten by values passed to params
        self.params = {'tolerance': 1e-4}
        self.reset(parameters)  
    def learn(self, Xtrain, ytrain):
        numsamples = Xtrain.shape[0]
        # init weights
        self.weights = np.zeros(len(Xtrain[0]))
        err = np.inf
        tolerance = self.params['tolerance']
        # precalculate XX and Xy
        dotXX = np.dot(Xtrain.T,Xtrain)/numsamples
        dotXy = np.dot(Xtrain.T,ytrain)/numsamples
        eta = 1/(2*Frobenius_norm(dotXX))
        c_w = 0
        # learn until err difference less than tolerance
        while (np.absolute(c_w-err) > tolerance):
            err = c_w
            self.weights = self.weights - eta*np.dot(dotXX,self.weights) + eta*dotXy
            c_w = np.sqrt(l2err_squared(np.dot(Xtrain,self.weights),ytrain))
        
    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        return ytest  
    

class SGDLinearRegression(Regressor):
    def __init__( self, parameters={} ):
        # Default parameters, any of which can be overwritten by values passed to params
        self.params = {'step_size': 0.01, 'epochs': 1000}
        self.reset(parameters)   

    def learn(self, Xtrain, ytrain):
        # set (self) parameters for later calculate
        self.Xtrain = Xtrain
        self.ytrain = ytrain
        self.ErrList = []
        self.ErrList_time = []        
        numsamples = Xtrain.shape[0]
        self.numsamples = numsamples
        self.weights = np.random.rand(len(Xtrain[0]))
        eta0 = self.params['step_size']
        # set time
        t0 = time.time()
        sec = 0.1
        # learn weights with the number of epochs (shuffle data before each epochs)
        for i in range(self.params['epochs']):
            Xtrain, ytrain = self.shuffle(Xtrain,ytrain,i)
            # learn weights for every sample
            for j in range(numsamples):
                g = np.dot(np.dot(Xtrain[j].T, self.weights)-ytrain[j], Xtrain[j])
                etat = eta0/(i+1)
                self.weights = self.weights - np.dot(etat,g)
            # get error for every epochs for learning curve
            self.ErrList.append(self.c())
            # get error for every second for learning curve
            if (time.time()-t0 >= sec):
                self.ErrList_time.append(self.c())
                sec += 0.1
        
    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        return ytest      
    
    #shuffle X and y by the same seed
    def shuffle(self, Xtrain,ytrain,seed):
        np.random.seed(seed)
        np.random.shuffle(Xtrain)
        np.random.seed(seed)
        np.random.shuffle(ytrain)  
        return (Xtrain, ytrain)
    
    # calculate c(w)
    def c(self):
        return l2err(np.dot(self.Xtrain,self.weights),self.ytrain)/(2*self.numsamples)    
    
    # return err list, each index represanting each epochs 
    def getErrList(self):
        return self.ErrList
    
    # return err list, each index represanting each second 
    def getErrList_time(self):
        return self.ErrList_time  
        
class BGDLinearRegression(Regressor):
    def __init__( self, parameters={} ):
        # Default parameters, any of which can be overwritten by values passed to params
        self.params = {'tolerance': 1e-4,'max_itration': 1e5,'tau':0.7}
        self.reset(parameters)  

    def learn(self, Xtrain, ytrain):
        # set (self) parameters for later calculate
        self.ErrList = []
        self.ErrList_time = []        
        numsamples = Xtrain.shape[0]
        self.weights = np.random.rand(len(Xtrain[0]))
        err = np.inf
        tolerance = self.params['tolerance'] 
        max_itration = self.params['max_itration'] 
        self.Xtrain = Xtrain
        self.ytrain = ytrain
        self.numsamples = numsamples
        c_w = self.c()
        itration = 0
        # set time
        t0 = time.time()
        sec = 0.1
        # learn weights until error difference less than tolarance or num of itration reach the max itration
        while (np.absolute(c_w-err) > tolerance and itration < max_itration):
            err = c_w
            g = np.dot(Xtrain.T, np.dot(Xtrain, self.weights)-ytrain )/self.numsamples
            # using line_search to get least eta value
            eta = self.line_search(g, tolerance, c_w, max_itration)
            self.weights = self.weights - eta*g
            c_w = self.c()
            itration += 1
            # get error for every epochs for learning curve
            self.ErrList.append(c_w)
            # get error for every second for learning curve
            if (time.time()-t0 >= sec):
                self.ErrList_time.append(self.c())
                sec += 0.1
        #print (itration)
        
    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        return ytest  
    
    # healper function for BGD
    def line_search(self, g, tolerance, c_w, max_itration):
        # set params
        tau = self.params['tau']
        eta = 1.0
        copy_weights = self.weights
        obj = c_w
        i = 0
        # line_search for getting the smallest eta value
        while (i < max_itration):
            copy_weights = self.weights - eta*g
            c_w = l2err(np.dot(self.Xtrain,copy_weights),self.ytrain)/(2*self.numsamples)
            if (c_w < obj - tolerance):
                break
            eta = eta*tau
            i += 1
        # check the loop stoping condition then return
        if (i>= max_itration):
            return 0
        else:
            return eta
        
    # calculate c(w)  
    def c(self):
        return l2err(np.dot(self.Xtrain,self.weights),self.ytrain)/(2*self.numsamples)

    # return err list, each index represanting each epochs 
    def getErrList(self):
        return self.ErrList
    
    # return err list, each index represanting each second 
    def getErrList_time(self):
        return self.ErrList_time  


class RMSPropLinearRegression(Regressor):
    def __init__( self, parameters={} ):
        # Default parameters, any of which can be overwritten by values passed to params
        self.params = {'eta': 0.01,'epsilon': 0.01,'epochs':1000}
        self.reset(parameters)   
     
    def learn(self, Xtrain, ytrain):
        # set (self) parameters for later calculate
        self.ErrList = []
        self.ErrList_time = []           
        numsamples = Xtrain.shape[0]
        self.weights = np.random.rand(len(Xtrain[0]))
        self.Xtrain = Xtrain
        self.ytrain = ytrain     
        self.numsamples = numsamples
        eta = self.params['eta']
        epsilon = self.params['epsilon']
        # set time
        t0 = time.time()
        sec = 0.1        
        # learn weights with the number of epochs (shuffle data before each epochs)
        for i in range(self.params['epochs']):
            Xtrain, ytrain = self.shuffle(Xtrain,ytrain,i)
            g2 = np.square(np.dot(np.dot(Xtrain[0].T, self.weights)-ytrain[0], Xtrain[0]))
            self.ErrList.append(self.c())
            # learn weights for every sample
            for j in range(numsamples):
                g = np.dot(np.dot(Xtrain[j].T, self.weights)-ytrain[j], Xtrain[j])
                g2 = 0.9*g2 + 0.1*np.square(g)
                self.weights = self.weights - eta*g/(np.sqrt(g2 + epsilon))
            # get error for every second for learning curve
            if (time.time()-t0 >= sec):
                self.ErrList_time.append(self.c())
                sec += 0.1            
        
    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        return ytest      
    
    #shuffle X and y by the same seed
    def shuffle(self, Xtrain,ytrain,seed):
        np.random.seed(seed)
        np.random.shuffle(Xtrain)
        np.random.seed(seed)
        np.random.shuffle(ytrain)  
        return (Xtrain, ytrain) 
    
    # calculate c(w)
    def c(self):
        return l2err(np.dot(self.Xtrain,self.weights),self.ytrain)/(2*self.numsamples)     
    
    # return err list, each index represanting each epochs 
    def getErrList(self):
        return self.ErrList
    
    # return err list, each index represanting each second 
    def getErrList_time(self):
        return self.ErrList_time      
    
class AMSGradLinearRegression(Regressor):
    def __init__( self, parameters={} ):
        # Default parameters, any of which can be overwritten by values passed to params
        self.params = {'beta1': 0.1,'beta2': 0.5,'eta': 0.01,'epsilon': 0.01,'epochs':1000}
        self.reset(parameters)       
    def learn(self, Xtrain, ytrain):
        # set (self) parameters for later calculate
        numsamples = Xtrain.shape[0]
        self.weights = np.random.rand(len(Xtrain[0]))
        eta = self.params['eta']
        epsilon = self.params['epsilon']
        beta1 = self.params['beta1']
        beta2 = self.params['beta2']
        # learn weights with the number of epochs (shuffle data before each epochs)
        for i in range(self.params['epochs']):
            Xtrain, ytrain = self.shuffle(Xtrain,ytrain,i)
            m = np.zeros(len(Xtrain[0]))
            v = np.zeros(len(Xtrain[0]))
            v_hat = np.zeros(len(Xtrain[0]))
            # learn weights for every sample
            for j in range(numsamples):
                g = np.dot(np.dot(Xtrain[j].T, self.weights)-ytrain[j], Xtrain[j])
                m = beta1*m + (1-beta1)*g
                v = beta2*v + (1-beta2)*np.square(g)
                v_hat = np.maximum(v_hat,v)
                self.weights = self.weights - eta*m/(np.sqrt(v_hat)+epsilon)
                
    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        return ytest      
    
    #shuffle X and y by the same seed
    def shuffle(self, Xtrain,ytrain,seed):
        np.random.seed(seed)
        np.random.shuffle(Xtrain)
        np.random.seed(seed)
        np.random.shuffle(ytrain)  
        return (Xtrain, ytrain)     