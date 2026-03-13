import numpy as np
import os, pathlib, math

class DataLoader(object):
    def __init__(self, shot, xdim, ydim, zdim):

        self.shot = shot
        self.xdim, self.ydim, self.zdim = xdim, ydim, zdim
        
    def get_train_dataset(self, inp, lab, num_episode=1, num_query=1):

        tdim = len(inp)

        em_inp = np.zeros((num_episode, self.shot+num_query, self.xdim, self.ydim, self.zdim))
        em_lab = np.zeros((num_episode, self.shot+num_query, 1))

        for j in range(num_episode):

            rd = np.random.choice(tdim, self.shot+num_query, replace=False)

            em_inp[j] = inp[rd]
            em_lab[j] = lab[rd].reshape(-1,1)

        query_inp = em_inp[:,:num_query]
        query_lab = em_lab[:,:num_query]

        support_inp = em_inp[:,num_query:]
        support_lab = em_lab[:,num_query:]

        return query_inp, query_lab, support_inp, support_lab

    def get_test_dataset(self, test_inp, test_lab, train_inp, train_lab):
        
        tdim = len(test_inp)
        
        query_inp = test_inp.reshape(-1, 1, self.xdim, self.ydim, self.zdim)
        query_lab = test_lab.reshape(-1, 1)
        
        support_inp = np.zeros((tdim, self.shot, self.xdim, self.ydim, self.zdim))
        support_lab = np.zeros((tdim, self.shot))
        
        for j in range(tdim):
            
            rd = np.random.choice(len(train_inp), self.shot, replace=False)
            
            support_inp[j] = train_inp[rd].reshape(self.shot, self.xdim, self.ydim, self.zdim)
            support_lab[j] = train_lab[rd].reshape(self.shot)
            
        return query_inp, query_lab, support_inp, support_lab
