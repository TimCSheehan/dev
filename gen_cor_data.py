# num_data_pnts = number of experimental trials
# mu0, mu1 = 2 element np.array that sets the mean of each variable in each condition
# cov0, cov1 = symetric 2 x 2 matrix with main diag specifying the variance of each variable 
# in each condition, and the off diag elements specifying the covariance

def gen_cor_data(num_data_pnts, mu0, mu1, cov0, cov1, plot):

    # number of variables, in this case lets keep it at 2 because that makes it easy to 
    # visualize
    V = 2

    # means of each variable in each condition
    mean_of_data0 = mu0; 
    mean_of_data1 = mu1 

    # generate some random data vectors drawn from normal
    data0 = np.random.randn(num_data_pnts,V) 
    data1 = np.random.randn(num_data_pnts,V) 

    # compute the eigenvalues and eigenvectors of the cov matrix
    evals0, evecs0 = eigh(cov0)
    evals1, evecs1 = eigh(cov1)

    # Construct c, so c*c^T = cov.
    c0 = np.dot(evecs0, np.diag(np.sqrt(evals0)))
    c1 = np.dot(evecs1, np.diag(np.sqrt(evals1)))

    # convert the data using by multiplying data by c
    # to be consistent with previous tutorials, we want the data running down columns...so do the double .T
    cdata0 = np.dot(c0, data0.T).T

    # then add in the mu offset...use np.ones * each condition mean
    cdata0 += np.hstack((np.ones((num_data_pnts,1))*mean_of_data0[0], np.ones((num_data_pnts,1))*mean_of_data0[1]))

    # repeat for the data from the second experimental condition 
    cdata1 = np.dot(c1, data1.T).T
    cdata1 += np.hstack((np.ones((num_data_pnts,1))*mean_of_data1[0], np.ones((num_data_pnts,1))*mean_of_data1[1])) 

    if plot:
        # plot the data...
        plt.scatter(cdata0[:,0], cdata0[:,1], color='b')
        plt.scatter(cdata1[:,0], cdata1[:,1], color='r')
        plt.xlabel('Resp Neuron 1', **fig_font)
        plt.ylabel('Resp Neuron 2', **fig_font)
        plt.legend(['Condition 1', 'Condition 2'])
        plt.show()
    
    return cdata0, cdata1