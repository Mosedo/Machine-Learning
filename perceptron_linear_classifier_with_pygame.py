import numpy as np
import math
import random

from numpy import random
import pygame
import sys

running=False

WIDTH=900
HEIGHT=600

#points=[[random.randint(WIDTH)]  for i in range(500)]

points=[]

colors=[]

draw_points=[[random.randint(WIDTH),random.randint(HEIGHT)]  for i in range(500)]

for p in draw_points:
    points.append([p[0]])
    if p[0] < WIDTH/2:
        colors.append((0,255,255))
    else:
        colors.append((255,255,255))

training_data=[
    
]

def normalize(x):
        return x/WIDTH

def drawPoint(x,y):
    # points.append((x,y))
    draw_points.append((x,y))

def drawCircle(x,y,color):
    pygame.draw.circle(window, color, (x,y), 5)

def drawLine():
    pygame.draw.line(window, (255,0,0), (WIDTH/2, 0), (WIDTH/2, HEIGHT),2)

for point in points:
    if point[0] <= WIDTH/2:
        training_data.append(
            {
                "input":[normalize(point[0])],
                "label":[1]
            }
        )
    else:
        training_data.append(
            {
                "input":normalize(point[0]),
                "label":[0]
            }
        )


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
        self.weights+=(np.matmul(error,np.array([input]))*self.learning_rate)
        self.bias+=(error*self.learning_rate)
        print(f"Error is {error[0][0]}")


perceptron=Perceptron()


for i in range(100):
    random.shuffle(training_data)
    for data in training_data:
        perceptron.train(data["input"],data["label"])

running=True

window=pygame.display.set_mode((WIDTH,HEIGHT))

while running:
    

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if pygame.mouse.get_pressed()[0]:
                if perceptron.feedFoward([normalize(pygame.mouse.get_pos()[0])]) < 0.5:
                    colors.append((255,255,255))
                    drawPoint(pygame.mouse.get_pos()[0],pygame.mouse.get_pos()[1])
                    
                else:
                    colors.append((0,255,255))
                    drawPoint(pygame.mouse.get_pos()[0],pygame.mouse.get_pos()[1])
                

    window.fill((0,0,0))

    drawLine()

    for index,key in enumerate(draw_points):
        drawCircle(draw_points[index][0],draw_points[index][1],colors[index])
    

    pygame.display.flip()

pygame.quit()
