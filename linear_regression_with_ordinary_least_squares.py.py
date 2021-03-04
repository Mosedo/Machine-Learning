import pygame

pygame.init()

HEIGHT=600
WIDTH=600

m=1
b=0

win=pygame.display.set_mode((HEIGHT,WIDTH))
pygame.display.set_caption("Linear Regression With Ordinary Least Squares")

run=True

#pygame.draw.circle(screen, (r,g,b), (x, y), R, w) #(r, g, b) is color, (x, y) is center, R is radius and w is the thickness of the circle border.

points=[]

def drawPoint(x,y):
    points.append((x,y))

def drawLine(x1,y1,x2,y2):
    x1=x1
    y1=(m*x1)+b
    x2=x2
    y2=(m*x2)+b
    pygame.draw.line(win, (255, 128, 0), (x1, y1), (x2, y2),3)


def linearRegression():
    xsum=0
    ysum=0
    num=0
    den=0
    for point in points:
        xsum+=point[0]
        ysum+=point[1]
    
    xmean=xsum/len(points)
    ymean=ysum/len(points)

    for p in points:
        num+=(p[0]-xmean)*(p[1]-ymean)
        den+=((p[0]-xmean)**2)
    
    global m
    global b

    m=num/den
    b=ymean-(m*xmean)

while run:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run=False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if pygame.mouse.get_pressed()[0]:
                drawPoint(pygame.mouse.get_pos()[0],pygame.mouse.get_pos()[1])
    win.fill((51,51,51))
    if len(points) > 0:
        for point in points:
            pygame.draw.circle(win, (255,255,255), (point[0], point[1]), 4)

    if len(points) > 1:
        linearRegression()
        drawLine(0,HEIGHT,WIDTH,0)
    pygame.display.flip()

pygame.quit()
