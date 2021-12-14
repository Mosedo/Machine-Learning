import numpy as np
import math
import random

from numpy import random

training_data=[
    {
        "input":[0.67373],
        "label":[1]
    },
    {
        "input":[0.25],
        "label":[0]
    },
    {
        "input":[0.8],
        "label":[1]
    },
    {
        "input":[0.1],
        "label":[0]
    },
    {
        "input":[0.5],
        "label":[1]
    },
    {
        "input":[0.5667],
        "label":[1]
    },
    {
        "input":[0.4],
        "label":[0]
    },
    {
        "input":[0.4676],
        "label":[0]
    },
    {
        "input":[0.2],
        "label":[0]
    },
    {
        "input":[0.9],
        "label":[1]
    },
    {
        "input":[0.0],
        "label":[0]
    },
    {
        "input":[0.7],
        "label":[1]
    },
    {
        "input":[0.3],
        "label":[0]
    }
]

class Perceptron:
    def __init__(self):
        self.weights=random.normal(size=(1,1))
        self.bias=1
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
        #weight_delta=(np.dot(error,np.array([input])))*self.learning_rate
        self.weights+=(np.matmul(error,np.array([input]))*self.learning_rate)
        self.bias+=(error*self.learning_rate)
        print(f"Error is {error[0][0]}")


perceptron=Perceptron()

for i in range(1000):
    random.shuffle(training_data)
    for data in training_data:
        perceptron.train(data["input"],data["label"])

print(perceptron.feedFoward([0.5]))
print(perceptron.feedFoward([0.1]))
print(perceptron.feedFoward([0.6]))
print(perceptron.feedFoward([0.8]))
print(perceptron.feedFoward([0.3]))
print(perceptron.feedFoward([0.9]))
print(perceptron.feedFoward([0.0]))
