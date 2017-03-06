#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 02:22:30 2017

@author: saheli06
"""

from __future__ import division
import numpy as np
from collections import defaultdict
from operator import add
from operator import div
from operator import mul
import matplotlib.pyplot as plt

#Reading the input file
first_file = open('data1.txt')
 
for line in first_file:
    f_line = line.rstrip()
    fl.append(float(f_line))

no_clus = input("Enter number of Gaussian:")
no_clus = int(no_clus)
      
mean=[]
sd =[]
prob_gauss = defaultdict(list) #P(b/Xi)
weights = [] #P(a),P(b)
prob_point=defaultdict(list) #P(Xi/b)P(b) where b is any gaussian
log_like=[]

np.random.seed(10)
def initialization(no_clus):
    mean = [np.random.choice(fl) for i in range(no_clus)]
    sg_sq = [np.random.randint(10, 30) for i in range(no_clus)]
    weights = [1 / no_clus for i in range(no_clus)]
    print "Initial Mean,Standard Deviation and Weight", mean,sg_sq,weights
    return mean,sg_sq,weights

def expectation(mean,sg_sq,weights):
    loglikelihood = []
    for j in range(no_clus):
        prob_point[j] = [(1 / np.sqrt(2*3.14 * sg_sq[j] ** 2.0)) * (np.exp(-(((val - mean[j]) ** 2) / (2.0 * sg_sq[j] ** 2))))*weights[j] for val in fl]
    prob_denominator=[0.0 for i in range(len(fl))]
    print"Probability of P(Xi/b)P(b) is:",prob_point
    for i in range(no_clus):
        prob_denominator=np.add(prob_denominator,np.add(prob_point[i],0))
    for i in range(no_clus):
        prob_gauss[i]=prob_point[i]/prob_denominator
    
    for k in range(len(fl)):
        temp = 0
        for j in range(no_of_gaussian):
            temp = temp + prob_point[j][k]
        loglikelihood.append(np.log(temp))
    log_like.append(sum(loglikelihood))
    print"Log Likelikhood is", log_like
    return prob_gauss
    
def maximization(prob_gauss):
    for i in range(no_clus):
        sums = 0
        difference = 0
        for j in range(len(fl)):
            sums = sums+(prob_gauss[i][j]*fl[j])
        mean[i] = sums/np.sum(prob_gauss[i])
        for j in range(len(fl)):
            difference = difference+(((fl[j]-mean[i])**2)*(prob_gauss[i][j]))
        sg_sq[i] = np.sqrt(difference/np.sum(prob_gauss[i]))
    weights = [sum(prob_gauss[i]) / len(prob_gauss[i]) for i in range(no_clus)]
    return mean,sg_sq,weights
    

mean,sg_sq,weights=initialization(no_clus)

for i in range(500):
    prob_gauss=expectation(mean,sg_sq,weights)
    print "P(b/Xi) of %d"%i, prob_gauss
    mean,sg_sq,weights=maximization(prob_gauss)
    if(i>2):
        if (((log_like[-1] - log_like[-2])/log_like[-2]) < 0.00000001): #Terminating condition for convergence
            break