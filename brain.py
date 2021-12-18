import numpy as np
import math
import random

from numpy import random


class Brain:
    def __init__(self):
        self.weights=np.array([[74.23268683]])
        self.bias=np.array([[-36.81015726]])
        self.learning_rate=0.1
    
    def feedFoward(self,input):
        result=np.dot(np.array([input]),self.weights)
        activated=[self.sigmoid(result.flatten()[0]+self.bias)]
        return np.array([activated])

    def sigmoid(self,x):
        return 1/(1+math.exp(-x))
    
    def train(self,input,label):
        result=self.feedFoward(input)
        error=np.subtract(np.array([label]),result)
        self.weights+=(np.matmul(error,np.array([input]))*self.learning_rate)
        self.bias+=(error*self.learning_rate)
        print(f"Error is {error[0][0]}")

