import pygame
from pygame import Surface
import sys
import time
import os
import smbus
import time  
from pygame.locals import QUIT,Rect
import serial
import select
import sys

address = 0x48
A0 = 0X40
bus = smbus.SMBus(1)

os.environ['SDL_VIDEO_WINDOW_POS']=str('0,-10')
pygame.init()
pygame.display.set_caption('Control panel of pallet hall recogntion system')
BLACK=(0,0,0)
RED=(255,0,0)
GREEN=(0,255,0)
BLUE=(0,0,255)
ORANGE=(255,165,0)
WHITE=(255,255,255)

def monitor_init():
    global screen, sub1, sub2, sub3, sub4, sub5
    screen=pygame.display.set_mode([800,600])
    # Image(Surface) which will be refrenced
    monitor = pygame.Surface((800, 600))

    # Monitor rectangles for sections of  the monitor
    mon1 = pygame.Rect(0,0,400,250)
    mon2 = pygame.Rect(400,0,400,250)
    mon3 = pygame.Rect(0,250,400,250)
    mon4 = pygame.Rect(400,250,400,250)
    mon5 = pygame.Rect(0,500,800,100)

    # subsurfaces of monitor
    # Note that subx needs refreshing when monitor changes.
    #pygame.draw.rect(screen, RED, [0,0,400,300])
    sub1 = monitor.subsurface(mon1)
    sub2 = monitor.subsurface(mon2)
    sub3 = monitor.subsurface(mon3)
    sub4 = monitor.subsurface(mon4)
    sub5 = monitor.subsurface(mon5)

    # Drawing a line on each split "screen"
    pygame.draw.rect(sub1, RED, [0,0,400,250])
    pygame.draw.rect(sub2, ORANGE, [0,0,400,250])
    pygame.draw.rect(sub3, GREEN, [0,0,400,250])
    pygame.draw.rect(sub4, BLUE, [0,0,400,250])
    pygame.draw.rect(sub5, BLACK, [0,0,800,100])

    # monitor1's view  to the top left corner
    screen.blit(sub1, (0,0))
    info1=pygame.font.SysFont(None,70)
    char=str("dist = ")
    disp1=info1.render(char,True,BLACK)
    screen.blit(disp1,(20,80))
    pygame.display.update()
    # monitor2's view is in the top right corner
    screen.blit(sub2, (400, 0))
    info2=pygame.font.SysFont(None,80)
    char=str("center : ")
    disp2=info2.render(char,True,BLACK)
    screen.blit(disp2,(450,75))     
    pygame.display.update()
    # monitor3's view is in the bottom left corner
    screen.blit(sub3, (0, 250))
    info3=pygame.font.SysFont(None,65)
    char1=str("center x= ")
    char2=str('center y= ')
                
    disp31=info3.render(char1,True,BLACK)
    disp32=info3.render(char2,True,BLACK)
    screen.blit(disp31,(20,300))
    screen.blit(disp32,(20,380))
    pygame.display.update()
    # monitor4's view is in the bottom right corner
    screen.blit(sub4, (400, 250))
    info4=pygame.font.SysFont(None,80)
    char=str("arrival : ")
    disp4=info4.render(char,True,BLACK)
    screen.blit(disp4,(450,350))      
    pygame.display.update()
    # monitor5's view is in the bottom 
    screen.blit(sub5, (0, 500))
    info5=pygame.font.SysFont(None,70)
    char=str("speed = ")
    disp5=info5.render(char,True,BLACK)
    screen.blit(disp5,(200,520))       
    pygame.display.update()
    # Update the screen
    #pygame.display.update()
    return

monitor_init()

def display(dist, center_flag, center_x, center_y, arrival_flag, speed):
    screen.blit(sub1, (0,0))
    info11=pygame.font.SysFont(None,70)
    dist=str("dist = ")+str(dist)+str(" mm")
    disp11=info11.render(dist,True,BLACK)
    screen.blit(disp11,(20,80))
    pygame.display.update()

    screen.blit(sub2, (400,0))
    info12=pygame.font.SysFont(None,80)
    if center_flag==False:
        dist21=str("center : ")
        disp2=info12.render(dist21,True,BLACK)
        screen.blit(disp2,(450,75))
    else:
        dist22=str("confirmed")    
        disp2=info12.render(dist22,True,BLACK)
        screen.blit(disp2,(450,75))  
    pygame.display.update()
    
    screen.blit(sub3, (0,250))
    info3=pygame.font.SysFont(None,65)
    dist31=str("center x= ")+str(center_x)+str(" mm")
    dist32=str('center y= ')+str(center_y)+str(" mm")              
    disp31=info3.render(dist31,True,BLACK)
    disp32=info3.render(dist32,True,BLACK)
    screen.blit(disp31,(20,300))
    screen.blit(disp32,(20,380))
    pygame.display.update()
   
    screen.blit(sub4, (400,250))
    if arrival_flag==False:
        info41=pygame.font.SysFont(None,80)
        dist41=str("arrival : ")
        disp4=info41.render(dist41,True,BLACK)
        screen.blit(disp4,(450,350))
    else:
        info42=pygame.font.SysFont(None,80)
        dist42=str("confirmed")
        disp4=info42.render(dist42,True,BLACK)
        screen.blit(disp4,(450,350))      
    pygame.display.update()
    
    screen.blit(sub5, (0,500))
    info51=pygame.font.SysFont(None,70)
    dist51=str("speed = ")+str(speed)+str(" mm/s")
    disp5=info51.render(dist51,True,WHITE)
    screen.blit(disp5,(200,520))     
    pygame.display.update()

    return

#display(2000, False, 5, 3, False, 200)
#time.sleep(5)
#display(1000, True, 3, 1, False, 150)
#time.sleep(5)
#display(500, True, 2, 0, True, 100)

while True:  
    ser = serial.Serial(
    port='/dev/ttyUSB0', 
    baudrate = 19200,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    bytesize=serial.EIGHTBITS,
    timeout=0.01
    )
    
    #print("Loop until press 'Ctrl+C' or Receive 'exit' from serial port")
    val2=100
    v=300
    display(val2, True, 2, 0, False, v)
    while val2 > 40:
        bus.write_byte(address,A0)
        value = bus.read_byte(address)
        #v = abs(int((value-240)))*10
        v = abs(int((value-238)))*10
        if v <= 20 :
            v = 0
        else:
            v = abs(int((value-238)))*10
        
        time.sleep(0.05)
    
        ser.write('F'.encode('utf-8'))
        s_rcv = ser.readline().rstrip(b'\r').decode('utf-8')

        if len(s_rcv)>8 :
             #print(s_rcv)
             val=list(s_rcv)
             val2=int(val[2])*1000+int(val[4])*100+int(val[5])*10+int(val[6])*1  
             #print(val2)
             display(val2, True, 2, 0, False, v)
        else:
             pass
                
    ser.close()
    display(val2, True, 2, 0, True, v)
    break

     
