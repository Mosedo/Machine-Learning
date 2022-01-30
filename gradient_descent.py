import numpy as np
import math
import random
import matplotlib.pyplot as plt

inputs=[np.array([1]),np.array([0])]
expected_vs=[1,0]
lr=0.08


def sigmoid(x):
    return 1 / (1 + math.exp(-x)) 

#input_hidden_weight=np.array([0.3])

input_hidden_weight=np.random.normal(size=(1,1))

bias=np.array([1.0])

errors_list=[]
losses_list=[]
gradients_list=[]

def train(input,expected):
    global input_hidden_weight
    global bias
    global errors_list
    global losses_list
    output=np.dot(input_hidden_weight,input)+bias
    output=sigmoid(output)

    error=expected-output

    loss=-expected*math.log10(output)+(1-expected)*math.log10(1-output)

    gradient=error*input*lr

    input_hidden_weight+=gradient
    bias+=error*lr

    losses_list.append(abs(loss))
    errors_list.append(abs(error))
    gradients_list.append(gradient)

    return f"Loss {loss} Error {error}"

def feedFoward(input):
    output=np.dot(input_hidden_weight,input)+bias
    output=sigmoid(output)
    return output

for epoch in range(5000):
    idx=random.randint(0,1)
    print(train(inputs[idx],expected_vs[idx]))

print("****************************")
print(feedFoward(np.array([1])))
print(feedFoward(np.array([0])))
print(feedFoward(np.array([1])))
print(feedFoward(np.array([0])))
print(feedFoward(np.array([1])))
print(feedFoward(np.array([1])))

# plt.plot(errors_list, label='Error')
# plt.plot(losses_list, label='Loss')
plt.plot(gradients_list, label='Gradient')
plt.xlabel('Epoch')
plt.ylabel('Error/Loss')
plt.legend(loc='upper right')
plt.show()



