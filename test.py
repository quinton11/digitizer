import pygame
import tensorflow as tf
#import os
from tkinter import *
import tkinter.messagebox


height = 600
width = 600
green = (0,255,0)
white = (255,255,255)
black = (0,0,0)
window = pygame.display.set_mode((height, width))
window.fill(black)

pygame.display.update()

model = tf.keras.models.load_model('mnist_ann.h5')


def wind_popup(msg):
    root = Tk()
    tkinter.messagebox.showinfo('Predict digit',
                               msg)
    root.mainloop()

def save_img():
    Img = pygame.transform.scale(window,(28,28))
    pygame.image.save(Img,'test.jpg')

def predict(model,window):
    #model = tf.keras.models.load_model('C:/quinton/pygames/mnist_ann.h5')
    
    #Loads image as a grayscale tensorflow img
    test = tf.keras.preprocessing.image.load_img('test.jpg',
                                                 color_mode = 'grayscale')

    #test = tf.image.rgb_to_grayscale(save_img())
    
    test_arr = tf.keras.preprocessing.image.img_to_array(test)
    
    #Resizes from a 3d image to a 2d one
    test_arr = test_arr[:,:,0]

    test_arr = test_arr/255

    r = model.predict(tf.expand_dims(test_arr,0)).argmax(axis = 1)
    
    wind_popup('You typed a {}'.format(int(r)))
    #print('You typed a {}'.format(int(r)))
    
    window.fill(black)
    pygame.display.update()




def gameloop():
    global model
    running = True
    while running:
        pygame.event.get()
    
        for events in pygame.event.get():
            if events.type == pygame.QUIT:
                running = False
            if events.type == pygame.KEYDOWN:
                if events.key == pygame.K_SPACE:
                    #running = False
                    save_img()
                    predict(model,window)
    
        keys = list(pygame.mouse.get_pressed())
        if keys[0]:
            x,y = pygame.mouse.get_pos()
            print('({},{})'.format(x,y))
            pygame.draw.rect(window,white,[x,y,40,40])
            pygame.display.update()

    
gameloop()           
#Img = pygame.transform.scale(window,(28,28))
#pygame.image.save(Img,'test.jpg')
pygame.quit()
#quit()


