#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 12:30:15 2017

@author: saheli06
"""

import numpy as np
import random as ran
import pylab as pl
ran.seed(100)

x = np.array(([0,0,0,0],[0,0,0,1],[0,0,1,0],[0,0,1,1],[0,1,0,0],[0,1,0,1],[0,1,1,0],[0,1,1,1],[1,0,0,0],[1,0,0,1],[1,0,1,0],[1,0,1,1], [1,1,0,0],[1,1,0,1],[1,1,1,0],[1,1,1,1]), dtype=float)
y= np.array(([0],[1],[1],[0],[1],[0],[0],[1],[1],[0],[0],[1],[0],[1],[1],[0]),dtype=float )
       
class Neural_Network(object):
    def __init__(self):
        #initializing the parameters
        self.limits = [-1,1]
         #weight parameters intialization
        self.w1 = [ran.random()*ran.choice(self.limits),ran.random()*ran.choice(self.limits),ran.random()*ran.choice(self.limits),ran.random()*ran.choice(self.limits)]
        self.w2 = [ran.random()*ran.choice(self.limits),ran.random()*ran.choice(self.limits),ran.random()*ran.choice(self.limits),ran.random()*ran.choice(self.limits)]
        self.w3 = [ran.random()*ran.choice(self.limits),ran.random()*ran.choice(self.limits),ran.random()*ran.choice(self.limits),ran.random()*ran.choice(self.limits)]
        self.w4 = [ran.random()*ran.choice(self.limits),ran.random()*ran.choice(self.limits),ran.random()*ran.choice(self.limits),ran.random()*ran.choice(self.limits)]
        self.w5 = [ran.random()*ran.choice(self.limits),ran.random()*ran.choice(self.limits),ran.random()*ran.choice(self.limits),ran.random()*ran.choice(self.limits)]
        self.b1 = [ran.random()*ran.choice(self.limits),ran.random()*ran.choice(self.limits),ran.random()*ran.choice(self.limits),ran.random()*ran.choice(self.limits)]
        self.b2 = ran.random()*ran.choice(self.limits)
        
    def sigmoid(self, z):
        return 1/(1+np.exp(-z))
    
    def forward_backward(self, x,y):
        #Initializing all the values
        er = 1
        n = 1
        alpha = 0.9
        epochs = 1
        eeta = 0.5
        #eeta = 0.15/(1-alpha) # Adding Momentum
        while er != 0:
            x_in = x[(epochs-1)%16]
            count = 0
            for i in x_in:
                if i == 1:
                    count+=1
            if count%2 == 0:
                y_out = 0
            else:
                y_out = 1
            ct = 0
            ct1 = 0
            #Forwardpropagation
            for j in x:
                self.z1 = (self.w1[0]*j[0]) + (self.w1[1]*j[1]) + (self.w1[2]*j[2]) + (self.w1[3]*j[3]) + self.b1[0]
                self.a1 = self.sigmoid(self.z1)
                self.z2 = (self.w2[0]*j[0]) + (self.w2[1]*j[1]) + (self.w2[2]*j[2]) + (self.w2[3]*j[3]) + self.b1[1]
                self.a2 = self.sigmoid(self.z2)
                self.z3 = (self.w3[0]*j[0]) + (self.w3[1]*j[1]) + (self.w3[2]*j[2]) + (self.w3[3]*j[3]) + self.b1[2]
                self.a3 = self.sigmoid(self.z3)
                self.z4 = (self.w4[0]*j[0]) + (self.w4[1]*j[1]) + (self.w4[2]*j[2]) + (self.w4[3]*j[3]) + self.b1[3]
                self.a4 = self.sigmoid(self.z4)
                self.z5 = (self.w5[0]*self.a1) + (self.w5[1]*self.a2) + (self.w5[2]*self.a3) + (self.w5[3]*self.a4) + self.b2
                self.a5 = self.sigmoid(self.z5)
                if abs(self.a5-y[ct1])<0.05:
                    ct+=1
                ct1+=1 
            if ct == 9: 
                er = 0
        	#Backpropagation 
            if er != 0:
                self.y1 = (self.w1[0]*x_in[0]) + (self.w1[1]*x_in[1]) + (self.w1[2]*x_in[2]) + (self.w1[3]*x_in[3]) + self.b1[0]
                self.a_y1 = self.sigmoid(self.y1)
                self.y2 = (self.w2[0]*x_in[0]) + (self.w2[1]*x_in[1]) + (self.w2[2]*x_in[2]) + (self.w2[3]*x_in[3]) + self.b1[1]
                self.a_y2 = self.sigmoid(self.y2)
                self.y3 = (self.w3[0]*x_in[0]) + (self.w3[1]*x_in[1]) + (self.w3[2]*x_in[2]) + (self.w3[3]*x_in[3]) + self.b1[2]
                self.a_y3 = self.sigmoid(self.y3)
                self.y4 = (self.w4[0]*x_in[0]) + (self.w4[1]*x_in[1]) + (self.w4[2]*x_in[2]) + (self.w4[3]*x_in[3]) + self.b1[3]
                self.a_y4 = self.sigmoid(self.y4)
                self.y5 = (self.w5[0]*self.a_y1) + (self.w5[1]*self.a_y2) + (self.w5[2]*self.a_y3) + (self.w5[3]*self.a_y4) + self.b2
                self.a_y5 = self.sigmoid(self.y5)
                
                for i in range(len(self.w1)):
                    self.w1[i] = self.w1[i] + (eeta*(y_out-self.a_y5)*self.a_y5*(1-self.a_y5)*self.w5[0]*self.a_y1*(1-self.a_y1)*x_in[i])
                    self.w2[i] = self.w2[i] + (eeta*(y_out-self.a_y5)*self.a_y5*(1-self.a_y5)*self.w5[1]*self.a_y2*(1-self.a_y2)*x_in[i])
                    self.w3[i] = self.w3[i] + (eeta*(y_out-self.a_y5)*self.a_y5*(1-self.a_y5)*self.w5[2]*self.a_y3*(1-self.a_y3)*x_in[i])
                    self.w4[i] = self.w4[i] + (eeta*(y_out-self.a_y5)*self.a_y5*(1-self.a_y5)*self.w5[3]*self.a_y4*(1-self.a_y4)*x_in[i])
                self.w5[0] = self.w5[0] + (eeta*(y_out-self.a_y5)*self.a_y5*(1-self.a_y5)*self.a_y1)
                self.w5[1] = self.w5[1] + (eeta*(y_out-self.a_y5)*self.a_y5*(1-self.a_y5)*self.a_y2)
                self.w5[2] = self.w5[2] + (eeta*(y_out-self.a_y5)*self.a_y5*(1-self.a_y5)*self.a_y3)
                self.w5[3] = self.w5[3] + (eeta*(y_out-self.a_y5)*self.a_y5*(1-self.a_y5)*self.a_y4)
                epochs+=1
        print "Epoch", epochs
        return self.w1, self.w2, self.w3, self.w4, self.w5
                
    def backward_update(self, x,y):
        #Weights Update
        self.w1, self.w2, self.w3, self.w4, self.w5 = NN.forward_backward(x,y)
        # Updating the values for prediction
        print "Output"
        for k in x:
            self.z11 = (self.w1[0]*k[0]) + (self.w1[1]*k[1]) + (self.w1[2]*k[2]) + (self.w1[3]*k[3]) + self.b1[0]
            self.a11 = self.sigmoid(self.z11)
            self.z22 = (self.w2[0]*k[0]) + (self.w2[1]*k[1]) + (self.w2[2]*k[2]) + (self.w2[3]*k[3]) + self.b1[1]
            self.a22 = self.sigmoid(self.z22)
            self.z33 = (self.w3[0]*k[0]) + (self.w3[1]*k[1]) + (self.w3[2]*k[2]) + (self.w3[3]*k[3]) + self.b1[2]
            self.a33 = self.sigmoid(self.z33)
            self.z44 = (self.w4[0]*k[0]) + (self.w4[1]*k[1]) + (self.w4[2]*k[2]) + (self.w4[3]*k[3]) + self.b1[3]
            self.a44 = self.sigmoid(self.z44)
            self.z55 = (self.w5[0]*self.a11) + (self.w5[1]*self.a22) + (self.w5[2]*self.a33) + (self.w5[3]*self.a44) + self.b2
            self.a55 = self.sigmoid(self.z55)
            print self.a55
            
NN = Neural_Network()
NN.backward_update(x,y)   
    