from __future__ import division  # floating point division
import csv
import random
import math
import numpy as np

import dataloaderFP as dtl
import regressionalgorithmsFP as algs
import matplotlib.pyplot as plt

def l2err(prediction,ytest):
    """ l2 error (i.e., root-mean-squared-error) """
    return np.linalg.norm(np.subtract(prediction,ytest))

def l1err(prediction,ytest):
    """ l1 error """
    return np.linalg.norm(np.subtract(prediction,ytest),ord=1)

def l2err_squared(prediction,ytest):
    """ l2 error squared """
    return np.square(np.linalg.norm(np.subtract(prediction,ytest)))

def geterror(predictions, ytest):
    # Can change this to other error values
    return 0.5*l2err(predictions,ytest)/ytest.shape[0]#/np.sqrt(ytest.shape[0])


# SGD: 'step_size': 0.01, 'epochs': 1000
# BGD: 'tolerance': 1e-4,'max_itration': 1e5,'tau':0.7
# RMSP: 'eta': 0.01,'epsilon': 0.01,'epochs':1000
def get_params(ind):
    #with best parameter
    if (ind==0):
        return (
        {'default': True},
                      )
    #test epochs = 100
    elif (ind==1):
        return (
        {'step_size': 0.1,'epochs': 100, 'tolerance': 1e-6, 'max_itration': 1e5, 'tau':0.6},
        {'step_size': 0.05,'epochs': 100, 'tolerance': 1e-7, 'max_itration': 1e5, 'tau':0.6},
        {'step_size': 0.01,'epochs': 100, 'tolerance': 1e-8, 'max_itration': 1e5, 'tau':0.6},
        {'step_size': 0.005,'epochs': 100, 'tolerance': 1e-9, 'max_itration': 1e5, 'tau':0.6},
        {'step_size': 0.001,'epochs': 100, 'tolerance': 1e-15, 'max_itration': 1e5, 'tau':0.6},
                      )
    #test epochs = 500
    elif (ind==2):
        return (
        {'step_size': 0.1,'epochs': 500, 'tolerance': 1e-6, 'max_itration': 1e5, 'tau':0.7},
        {'step_size': 0.05,'epochs': 500, 'tolerance': 1e-7, 'max_itration': 1e5, 'tau':0.7},
        {'step_size': 0.01,'epochs': 500, 'tolerance': 1e-8, 'max_itration': 1e5, 'tau':0.7},
        {'step_size': 0.005,'epochs': 500, 'tolerance': 1e-9, 'max_itration': 1e5, 'tau':0.7},
        {'step_size': 0.001,'epochs': 500, 'tolerance': 1e-15, 'max_itration': 1e5, 'tau':0.7},
                      )
    #test epochs = 1000
    elif (ind==3):
        return (
        {'step_size': 0.1,'epochs': 1000, 'tolerance': 1e-6, 'max_itration': 1e5, 'tau':0.8},
        {'step_size': 0.05,'epochs': 1000, 'tolerance': 1e-7, 'max_itration': 1e5, 'tau':0.8},
        {'step_size': 0.01,'epochs': 1000, 'tolerance': 1e-8, 'max_itration': 1e5, 'tau':0.8},
        {'step_size': 0.005,'epochs': 1000, 'tolerance': 1e-9, 'max_itration': 1e5, 'tau':0.8},
        {'step_size': 0.001,'epochs': 1000, 'tolerance': 1e-15, 'max_itration': 1e5, 'tau':0.8},
                      )
    #test epochs = 2000
    elif (ind==4):
        return (
        {'step_size': 0.1,'epochs': 2000, 'tolerance': 1e-6, 'max_itration': 1e5, 'tau':0.9},
        {'step_size': 0.05,'epochs': 2000, 'tolerance': 1e-7, 'max_itration': 1e5, 'tau':0.9},
        {'step_size': 0.01,'epochs': 2000, 'tolerance': 1e-8, 'max_itration': 1e5, 'tau':0.9},
        {'step_size': 0.005,'epochs': 2000, 'tolerance': 1e-9, 'max_itration': 1e5, 'tau':0.9},
        {'step_size': 0.001,'epochs': 2000, 'tolerance': 1e-15, 'max_itration': 1e5, 'tau':0.9},
                      )

    else:
        return

def get_regressionalgs(ind):
    #main function
    if (ind==0):
        return {
                'Random': algs.Regressor(),
                'SGDLinearRegression': algs.SGDLinearRegression({'step_size': 0.01, 'epochs': 1000}),
                'BGDLinearRegression': algs.BGDLinearRegression({'tolerance': 1e-6,'max_itration': 1e5,'tau':0.7}),
                'RMSPropLinearRegression': algs.RMSPropLinearRegression({'eta': 0.01,'epsilon': 0.01,'epochs':1000})
        }
    elif (ind==1):
        return {
            #'SGDLinearRegression': algs.SGDLinearRegression({'step_size': 0.01, 'epochs': 1000}),
            'BGDLinearRegression': algs.BGDLinearRegression({'tolerance': 1e-6,'max_itration': 1e5,'tau':0.7}),
            #'RMSPropLinearRegression': algs.RMSPropLinearRegression({'eta': 0.01,'epsilon': 0.01,'epochs':1000})
        }
    else:
        return



def cross_validate(K, X, Y, regressionalgs):
    # init samples per fold
    samples_per_fold = int(len(X)/K)
    # init error
    errors = {}
    total_errors = {}
    # for plot
    errPerEpochs = {}
    errPerSec = {}
    lenEE = {}
    lenES = {}
    plot_para = {}
    # init result
    best_algorithm = None
    best_error = None
    # get params
    parameters = get_params(4)
    # run loop
    for para in range(len(parameters)):
        errors[para] = np.zeros(K)
        total_errors[para] = 0
        errPerEpochs[para] = np.zeros(5000)
        errPerSec[para] = np.zeros(5000)
        lenEE[para] = 5000
        lenES[para] = 5000

        for k in range(K):
            # get disjoint test and training sets
            start_pos = int(samples_per_fold*k)
            end_pos = int(samples_per_fold*(k+1))
            test_x = X[start_pos:end_pos]
            test_y = Y[start_pos:end_pos]
            train_x = np.append(X[:start_pos],X[end_pos:len(X)],axis=0)
            train_y = np.append(Y[:start_pos],Y[end_pos:len(Y)],axis=0)
            for learnername, learner in regressionalgs.items():
                params = parameters[para]
                # Reset learner for new parameters
                learner.reset(params)
                print ('Running learner = ' + learnername + ' on parameters ' + str(learner.getparams()))
                #Train model
                learner.learn(train_x, train_y)
                #get plot data
                epochs_error = learner.getErrList()
                time_error = learner.getErrList_time()
                lenEE[para] = min(lenEE[para],len(epochs_error))
                lenES[para] = min(lenES[para],len(time_error))
                errPerEpochs[para][:lenEE[para]]  += epochs_error[:lenEE[para]]
                errPerSec[para][:lenES[para]]  += time_error[:lenES[para]]
                #print("------------",errPerEpochs)
                plot_para[para] = params['step_size']
                # Test model
                predictions = learner.predict(test_x)
                error = geterror(test_y, predictions)
                print ('Error for ' + learnername + ': ' + str(error))
                errors[para][k] = error
                total_errors[para] += error
                break
        best_algorithm = parameters[para]
        best_error = total_errors[para]
        print ("parameter:",best_algorithm,",with error:",best_error/K,"+-",str(np.std(errors[para])/math.sqrt(K)))

    #plot for training data
    plot_train = False
    if plot_train:
        plot_Epochs = False
        # plot errPerEpochs
        if plot_Epochs:
            print(errPerEpochs)
            for learnername in errPerEpochs:
                print (errPerEpochs[learnername])
                plt.plot(errPerEpochs[learnername][:lenEE[learnername]], label = 'step_size: '+str(plot_para[learnername]))
                plt.xlabel('epochs')
                plt.legend()
                plt.show()
        # plot errPerSec
        else:
            print(errPerEpochs,errPerSec)
            for learnername in errPerSec:
                print (errPerSec[learnername])
                plt.plot(errPerSec[learnername][:lenES[learnername]], label = 'step_size: '+str(plot_para[learnername]))
                plt.xlabel('sec/10')
                plt.legend()
                plt.show()
    #plot for testing data
    plot_test = True
    total_errors_list = []
    x = [1e-6,1e-7,1e-8,1e-9,1e-15]
    for i in range(len(parameters)):
        total_errors_list.append(total_errors[i]/K)
    print(total_errors_list,total_errors)
    if plot_test:
        dim = min(len(x),len(total_errors_list))
        plt.plot(x[:dim],total_errors_list[:dim])
        #plt.xticks([0.1,0.05,0.01,0.005,0.001])
        plt.xlabel('tolerance')
        plt.ylabel('everage error')
        #plt.legend()
        plt.show()

    # best parameter
    for para in range(len(parameters)):
        if (total_errors[para]<best_error):
            best_error = total_errors[para]
            best_algorithm = parameters[para]

    print ("best parameter:",best_algorithm,",with error:",best_error/K,"+-",str(np.std(errors[para])/math.sqrt(K)))

    return best_algorithm

if __name__ == '__main__':
    trainsize = 1000
    testsize = 5000
    numruns = 10
    k_fold = True

    regressionalgs = get_regressionalgs(0)
    if (k_fold):
        regressionalgs = get_regressionalgs(1)
        numruns = 1

    numalgs = len(regressionalgs)

    # Enable the best parameter to be selected, to enable comparison
    # between algorithms with their best parameter settings
    parameters = get_params(0)
    numparams = len(parameters)

    errors = {}
    for learnername in regressionalgs:
        errors[learnername] = np.zeros((numparams,numruns))

    errPerEpochs = {'SGDLinearRegression': [],'BGDLinearRegression': [],'RMSPropLinearRegression':[]}
    errPerSec = {'SGDLinearRegression': [],'BGDLinearRegression': [],'RMSPropLinearRegression':[]}

    if (k_fold):
        K = 5
        trainset, testset = dtl.load_train(trainsize,testsize)
        best_algorithm = cross_validate(K, trainset[0], trainset[1], regressionalgs)
    else:
        for r in range(numruns):
            trainset, testset = dtl.load_train(trainsize,testsize)
            print(('Running on train={0} and test={1} samples for run {2}').format(trainset[0].shape[0], testset[0].shape[0],r))

            for p in range(numparams):
                params = parameters[p]
                for learnername, learner in regressionalgs.items():
                    # Reset learner for new parameters
                    learner.reset(params)
                    print ('Running learner = ' + learnername + ' on parameters ' + str(learner.getparams()))
                    # Train model
                    learner.learn(trainset[0], trainset[1])
                    # Test model
                    predictions = learner.predict(testset[0])
                    error = geterror(testset[1], predictions)
                    if (learnername == 'SGDLinearRegression' or learnername == 'BGDLinearRegression' or learnername == 'RMSPropLinearRegression'):
                        errPerEpochs[learnername] = learner.getErrList()
                        errPerSec[learnername] = learner.getErrList_time()
                    print ('Error for ' + learnername + ': ' + str(error))
                    errors[learnername][p,r] = error
                print ("\n")


        for learnername in regressionalgs:
            besterror = np.mean(errors[learnername][0,:])
            bestparams = 0
            for p in range(numparams):
                aveerror = np.mean(errors[learnername][p,:])
                if aveerror < besterror:
                    besterror = aveerror
                    bestparams = p



            # find best stderr
            beststderr = np.std(errors[learnername][bestparams,:])/np.sqrt(numruns)

            # Extract best parameters
            learner.reset(parameters[bestparams])
            print ('Best parameters for ' + learnername + ': ' + str(parameters[bestparams]))
            print ('Average error for ' + learnername + ': ' + str(besterror))
            print ('Standard error for ' + learnername + ': ' + str(beststderr))
            print ('\n')

        # plot code
        plot = True
        if plot:
            plot_Epochs = False
            # plot errPerEpochs
            if plot_Epochs:
                for learnername in errPerEpochs:
                    print (errPerEpochs[learnername])
                    plt.plot(errPerEpochs[learnername], label = learnername)
                    plt.xlabel('epochs')
                    plt.legend()
                    plt.show()
            # plot errPerSec
            else:
                for learnername in errPerSec:
                    print (errPerSec[learnername])
                    plt.plot(errPerSec[learnername][:20], label = learnername)
                    plt.xlabel('sec/10')
                    plt.legend()
                    plt.show()
