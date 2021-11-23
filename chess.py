# Steering Behavior Examples
# Seek & Approach
# KidsCanCode 2016
# Video lesson: https://youtu.be/g1jo_qsO5c4

import pygame as pg
from random import randint, uniform
vec = pg.math.Vector2

WIDTH = 800
HEIGHT = 600
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
MAX_SPEED = 5
MAX_FORCE = 0.1
APPROACH_RADIUS = 120

screen = pg.display.set_mode((WIDTH, HEIGHT))

def drawCircle(x,y,color):
    pg.draw.circle(screen, color, (x,y), 5)

class Ball:
    def __init__(self):
        self.pos = vec(randint(0, WIDTH), randint(0, HEIGHT))
        self.vel = vec(MAX_SPEED, 0).rotate(uniform(0, 360))
        self.acc = vec(0, 0)
        
    
    def draw(self):
        pg.draw.circle(screen, (0,255,255), (self.pos.x,self.pos.y), 20)


    def follow_mouse(self):
        mpos = pg.mouse.get_pos()
        self.acc = (mpos - self.pos).normalize() * 0.5
    
    def seek(self, target):
        self.desired = (target - self.pos).normalize() * MAX_SPEED
        steer = (self.desired - self.vel)
        if steer.length() > MAX_FORCE:
            steer.scale_to_length(MAX_FORCE)
        return steer

    def seek_with_approach(self, target):
        self.desired = (target - self.pos)
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

    def update(self):
        #self.follow_mouse()
        self.acc = self.seek_with_approach(pg.mouse.get_pos())
        # equations of motion
        self.vel += self.acc
        if self.vel.length() > MAX_SPEED:
            self.vel.scale_to_length(MAX_SPEED)
        self.pos += self.vel

pg.init()

clock = pg.time.Clock()

ball=Ball()

running = True
while running:
    clock.tick(FPS)
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False

    screen.fill((0,0,0))

    ball.draw()
    ball.update()

    pg.display.flip()

pg.quit()
