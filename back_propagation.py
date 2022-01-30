import errno
import numpy as np
import math
import matplotlib.pyplot as plt

from sklearn.model_selection import learning_curve

weights_ih=np.random.normal(size=(1,1))
weights_ho=np.random.normal(size=(1,1))
bias_in=np.random.normal(size=(1,1))
bias_ho=np.random.normal(size=(1,1))
learning_rate=0.01
losses_list=[]

def feedFoward(input):
    hidden_inputs=np.dot(input,weights_ih)
    hidden_inputs=hidden_inputs+bias_in
    hidden_inputs=sigmoid(hidden_inputs)
    output=np.dot(hidden_inputs,weights_ho)
    output=output+bias_ho
    output=sigmoid(output)
    return output

def train(input,target):

    global weights_ho
    global weights_ih
    global bias_in
    global bias_ho
    global learning_rate

    hidden_inputs=np.dot(input,weights_ih)
    hidden_inputs=hidden_inputs+bias_in
    hidden_inputs=sigmoid(hidden_inputs)
    output=np.dot(hidden_inputs,weights_ho)
    output=output+bias_ho
    output=sigmoid(output)


    loss=-target[0]*math.log10(output[0][0])+(1-target[0])*math.log10(1-output[0][0])
    losses_list.append(abs(loss))

    #Calculate Errors
    error=target-output
    hidden_errors=np.dot(weights_ho.T,error)

    #Calculate gradients

    #Formulas
    #Hidden_error=transposed_hidden_weights dot error
    #hidden_gradients=lr*error*(output*(1-output)) dot H- hidden_input transposed
    #input_gradients=lr*hidden_error*(H*(1-H)) dot I - Input transposed


    

    hidden_gradient=learning_rate*error*dsigmoid(output)
    hidden_gradient=np.dot(hidden_gradient,hidden_inputs.T)
    input_gradient=learning_rate*hidden_inputs*dsigmoid(hidden_inputs)
    input_gradient=np.dot(input_gradient,np.array(input).T)

    weights_ho+=hidden_gradient
    weights_ih=input_gradient
    bias_ho+=learning_rate*error
    bias_in+=learning_rate*hidden_errors
 
    


    return f"Loss {loss} Error {error}"


def sigmoid(x):
    return 1/(1+np.exp(-x))

def dsigmoid(x):
    return x*(1-x)

for epoch in range(5000):
    print(train([1],[1]))

plt.plot(losses_list, label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()
