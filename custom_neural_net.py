import numpy as np
import math
import matplotlib.pyplot as plt
import random


losses_list=[]

data=[]
y=[]

WIDTH=200

for i in range(WIDTH):
    if i <= WIDTH/2:
        data.append({
            "input":[i/WIDTH],
            "target":[0]
        })
    else:
        data.append({
            "input":[i/WIDTH],
            "target":[1]
        })

class NeuralNetwork:
    def __init__(self,input_nodes,hidden_nodes,output_nodes):
        self.input_nodes=input_nodes
        self.hidden_nodes=hidden_nodes
        self.output_nodes=output_nodes
        self.weights_ih=np.random.normal(size=(self.hidden_nodes,self.input_nodes))
        self.weights_ho=np.random.normal(size=(self.output_nodes,self.hidden_nodes))
        self.bias_ih=np.array([1])
        self.bias_ho=np.array([1])
        self.learning_rate=0.01

    def feedFoward(self,input):
        hidden_inputs=np.dot(self.weights_ih,input)
        hidden_inputs=hidden_inputs+self.bias_ih
        hidden_inputs=self.sigmoid(hidden_inputs)
        output=np.dot(self.weights_ho,hidden_inputs)
        output=output+self.bias_ho
        output=self.sigmoid(output)
        return output
    
    def train(self,input,label):
        hidden_inputs=np.dot(self.weights_ih,input)
        hidden_inputs=hidden_inputs+self.bias_ih
        hidden_inputs=self.sigmoid(hidden_inputs)
        output=np.dot(self.weights_ho,hidden_inputs)
        output=output+self.bias_ho
        output=self.sigmoid(output)

        
        loss=-label[0]*math.log10(output[0])+(1-label[0])*math.log10(1-output[0])
        losses_list.append(abs(loss))

        #Calculate errors
        error=label-output[0]
        hidden_errors=np.dot(self.weights_ho.T,error)

        #gradients
        hidden_gradients=self.learning_rate*error*self.dsigmoid(output)
        hidden_gradients=np.dot(np.array([hidden_inputs]).T,np.array([hidden_gradients]))

        input_gradients=self.learning_rate*hidden_errors*self.dsigmoid(np.array([hidden_inputs]))
        input_gradients=np.dot(np.array([input]).T,input_gradients)
 
        

        self.weights_ho+=hidden_gradients.reshape(1,hidden_gradients.size)
        self.bias_ho=self.bias_ho+(self.learning_rate*error)
        self.weights_ih+=input_gradients.flatten().reshape(input_gradients.flatten().size,1)
        self.bias_ih=self.bias_ih+(self.learning_rate*hidden_errors)

        

        return f"Loss {loss} Error {error}"

        


    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
    
    def dsigmoid(self,output):
        return output*(1-output)
    
nn=NeuralNetwork(1,10,1)

#print(nn.train([1],[1]))

for epoch in range(500):
    random.shuffle(data)
    for x in data:
        print(nn.train(x["input"],x["target"]))


plt.plot(losses_list, label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()

print(nn.feedFoward([50/WIDTH]))
print(nn.feedFoward([150/WIDTH]))
print(nn.feedFoward([80/WIDTH]))
print(nn.feedFoward([180/WIDTH]))
print(nn.feedFoward([199/WIDTH]))
print(nn.feedFoward([110/WIDTH]))
print(nn.feedFoward([90/WIDTH]))