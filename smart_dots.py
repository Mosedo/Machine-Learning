import pygame
import math
import numpy as np
from numpy import random
import sys
import random
vec = pygame.math.Vector2
import brain


HEIGHT=600
WIDTH=900
FPS = 60
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
DARKGRAY = (40, 40, 40)

# Mob properties
MOB_SIZE = 32
MAX_SPEED = 2
MAX_FORCE = 0.1
APPROACH_RADIUS = 120

remove_list=[]

eaten=[]


print(brain.Brain().feedFoward([0.8]))
print(brain.Brain().feedFoward([0.6]))
print(brain.Brain().feedFoward([0.3]))
print(brain.Brain().feedFoward([0.7]))

win=pygame.display.set_mode((WIDTH,HEIGHT))

class Food:
    def __init__(self,index):
        self.position=vec(random.randint(10,WIDTH-10),random.randint(10,HEIGHT-10))
        self.color=(0,255,255)
        self.index=index
        self.code=random.uniform(0.5,1.0)
        self.removed=False
        self.size=4
    def draw(self):
        pygame.draw.circle(win, self.color, (self.position.x,self.position.y), self.size)

class Poison:
    def __init__(self,index):
        self.position=vec(random.randint(10,WIDTH-10),random.randint(10,HEIGHT-10))
        self.color=(255,0,0)
        self.index=index
        self.code=random.uniform(0,0.49)
        self.size=4
        self.removed=False
    def draw(self):
        pygame.draw.circle(win, self.color, (self.position.x,self.position.y), self.size)

foods=[]

for i in range(100):
    if i <= 60:
        foods.append(Food(i))
    else:
       foods.append(Poison(i)) 

class Agent:
    def __init__(self,idx):
        self.position=vec(random.randint(10,WIDTH-10),random.randint(10,HEIGHT-10))
        self.vel=vec(0,0)
        self.acceleration=vec(0,0)
        self.color=(0,255,0)
        self.index=idx
        self.brain=brain.Brain()
        self.random_directions=[(random.randint(10,WIDTH-10),random.randint(10,HEIGHT-10))]
        self.searching_food=True
        self.nearby_food=[]
        self.search_radius=100
        self.life=100
        self.not_food=[]
    
    def mapFood(self,f):
        if ((f.position.x > self.position.x) and f.position.x <= self.position.x+self.search_radius) or ((f.position.y > self.position.y) and f.position.y <= self.position.y+self.search_radius):
            if f is not None:
                return f

    def drawAgent(self):

        if(len(self.not_food) > 2 and self.life > 0):
            self.life-=5
            self.not_food.clear()
        
        if self.life <= 0:
            temp_index=self.index
            try:
                del agents[self.index]
                reorganizeList(temp_index)
            except IndexError:
                print("Passed")
                

        pygame.draw.circle(win, self.color, (self.position.x,self.position.y), 5)

        if self.life == 100:
            self.color=(0,255,255)
        elif self.life <= 75 and self.life >= 50:
            self.color=(255,165,0)
        elif self.life <= 50 and self.life >=25:
            self.color=(255,255,0)
        elif self.life <= 25 and self.life >=0:
            self.color=(255,0,0)
        elif self.life <= 0:
            self.color=(255,0,0)

    def follow_mouse(self):
        mpos = pygame.mouse.get_pos()
        self.acc = (mpos - self.pos).normalize() * 0.5
    
    def seek(self, target):
        self.desired = (target - self.pos).normalize() * MAX_SPEED
        steer = (self.desired - self.vel)
        if steer.length() > MAX_FORCE:
            steer.scale_to_length(MAX_FORCE)
        return steer

    def seek_with_approach(self, target):
        self.desired = (target - self.position)
        dist = self.desired.length()
        self.desired.normalize_ip()
        if dist < APPROACH_RADIUS:
            self.desired *= dist / APPROACH_RADIUS * MAX_SPEED
        else:
            self.desired *= MAX_SPEED
        steer = (self.desired - self.vel)
        if steer.length() > MAX_FORCE:
            steer.scale_to_length(MAX_FORCE)
        
        return steer
    
    def seek_food(self,target):
        self.acceleration = self.seek_with_approach(target)
        self.vel += self.acceleration
        if self.vel.length() > MAX_SPEED:
            self.vel.scale_to_length(MAX_SPEED)
        self.position += self.vel
    
    def seek_random(self,target):
        self.acceleration = self.seek_with_approach(target)
        self.vel += self.acceleration
        if self.vel.length() > MAX_SPEED:
            self.vel.scale_to_length(MAX_SPEED)
        self.position += self.vel

        if len(self.nearby_food) < 1:
            #if self.position.x-self.random_directions[0][0] <1 and self.position.y-self.random_directions[0][1] <1:
            if abs(self.random_directions[0][0]-self.position.x) < 3:
                self.random_directions[0]=(random.randint(10,WIDTH-10),random.randint(10,HEIGHT-10))
                found_food=list(map(self.mapFood,foods))
                found_food=[fd for fd in found_food if fd is not None and fd.removed is False]

                if len(found_food) > 0:
                    for ff in found_food:
                        self.nearby_food.append(ff)

                        if ff.code < 0.5:
                            self.nearby_food.append(ff)
                
        else:
            self.searching_food=False
        
        


        
    


            
        
        
    def search_food(self):
        
        if self.searching_food:
            self.seek_random(self.random_directions[0])
            
        else:
            if len(self.nearby_food) > 0:
                if abs(self.nearby_food[0].position.x-self.position.x) < 3:
                    try:

                        pred=self.brain.feedFoward(self.nearby_food[0].code)[0][0]

                        #print(pred)

                        if pred >=0.5:
                            foods[self.nearby_food[0].index].removed=True
                            foods[self.nearby_food[0].index].color=(0,0,0)
                            foods[self.nearby_food[0].index].size=0


                            if self.nearby_food[0].code < 0.5:
                                self.color=(255,255,0)
                            
                            if self.life < 100:
                                self.life+=2
                        # else:
                        #     self.not_food.append(1)
                        
                    except IndexError:
                        pass

                    
                        
                    del self.nearby_food[0]
                    


                else:
                    self.seek_food(self.nearby_food[0].position)
                    # if self.brain.feedFoward(self.nearby_food[0].code)[0][0]>=0.5: 
                    #     self.seek_food(self.nearby_food[0].position) 
                    # else:
                    #     del self.nearby_food[0]
                    #     self.random_directions[0]=(random.randint(10,WIDTH-10),random.randint(10,HEIGHT-10))
                    
            else:
                if len(self.nearby_food) > 0:
                    self.seek_food(self.nearby_food[0].position)
                else:
                    self.random_directions[0]=(random.randint(10,WIDTH-10),random.randint(10,HEIGHT-10))
                    self.searching_food=True
                    eaten.append(1)
            
        


    

agents=[Agent(i) for i in range(2)]

def addMoreFood():
    for i in range(100):
        if i <= 60:
            foods.append(Food(i))
        else:
            foods.append(Poison(i)) 


win=pygame.display.set_mode((WIDTH,HEIGHT))

testing=True

def deleteAllFoods():
    sum=0
    for food in foods:
        if food.removed is True:
            sum+=1
    
    if sum == len(foods):
        foods.clear()
        remove_list.clear()
        addMoreFood()
    
def addAFood():
    idx=foods[len(foods)-1].index+1
    foods.append(Food(idx))
    idx2=foods[len(foods)-1].index+1
    foods.append(Food(idx2))
    idx3=foods[len(foods)-1].index+1
    foods.append(Food(idx3))
    idx4=foods[len(foods)-1].index+1
    foods.append(Food(idx4))
    idx5=foods[len(foods)-1].index+1
    foods.append(Food(idx5))
    idx6=foods[len(foods)-1].index+1
    foods.append(Food(idx6))
    idx7=foods[len(foods)-1].index+1
    foods.append(Food(idx7))
    idx8=foods[len(foods)-1].index+1
    foods.append(Food(idx8))
    idx9=foods[len(foods)-1].index+1
    foods.append(Food(idx9))
    idx10=foods[len(foods)-1].index+1
    foods.append(Food(idx10))

def reorganizeList(index):
    for agent in agents:
        #print(f"{index}-------{agent.index}")
        if agent.index > index:
            agent.index-=1


clock = pygame.time.Clock()

while True:

    deleteAllFoods() 

    if len(eaten) ==3:
        addAFood()
        eaten.clear()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
    
    win.fill((0,0,0))
    
    for agent in agents:
        agent.drawAgent()
        
        if len(foods) > 0:
            agent.search_food()
            
    for food in foods:
        food.draw()
    

    
    pygame.display.flip()
    #clock.tick(60)
pygame.quit()
