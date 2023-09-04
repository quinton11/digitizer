import pygame
import numpy as np
import os
import tensorflow as tf


pygame.init()

height = 600
width = 800
green = (51,151,34)
light_green = (61,231,33)
white = (255,255,255)
black = (0,0,0)

font = pygame.font.Font(os.path.join('font','RobotoMono-Regular.ttf'),16)

button_text = font.render('Reset',True,black)
predict_text = font.render('Predict',True,black)


window = pygame.display.set_mode((width, height))
window.fill(black)

pygame.display.set_caption('Predict Digits')
pygame.display.update()

# x,y,w,h
write_area = [0,0,800,height-100]
reset_button = [width-100,530,80,25]
predict_button = [width-200,530,80,25]


model = tf.keras.models.load_model('mnist_ann.h5')


def save_img():
    Img = pygame.transform.scale(window,(28,28))
    pygame.image.save(Img,'test.jpg')

def get_patch():
    #grab screen pixels
    sub_surface = window.subsurface(pygame.Rect(*write_area))
    sub_surface = pygame.transform.scale(sub_surface,(28,28))
    #print(sub_surface.map_rgb(black))

    #format pixels
    arr = np.transpose(np.array(pygame.surfarray.array2d(sub_surface)))
    arr[arr == sub_surface.map_rgb(black)] = 0
    arr[arr == sub_surface.map_rgb(white)] = 255
    return arr

def predict(model,window,patch):
    patch = patch/255

    r = model.predict(tf.expand_dims(patch,0)).argmax(axis = 1)
    #print(r)
    predict_text = font.render('Looks like a {}'.format(int(r)),True,white)

    window.blit(predict_text,(30,530))
    pygame.display.update()




def gameloop():
    global model
    edit=False
    running = True
    try:
        while running:
            pygame.event.get()
        
            for events in pygame.event.get():
                if events.type == pygame.QUIT:
                    running = False
        
            keys = list(pygame.mouse.get_pressed())
            mx,my = pygame.mouse.get_pos()

            if keys[0]:

                # Check if mouse is in predict_button and edit=true
                if predict_button[0]<= mx <= predict_button[0]+predict_button[2] and predict_button[1] <= my <= predict_button[1]+predict_button[3]:
                    if edit:
                        predict(model,window,get_patch())
                        edit=False


                if my<write_area[3] - 40:
                    edit=True
                    pygame.draw.rect(window,white,[mx,my,40,40])
                if reset_button[0]<= mx <= reset_button[0]+reset_button[2] and reset_button[1] <= my <= reset_button[1]+reset_button[3]:
                    # clear screen
                    window.fill(black)
                
                pygame.display.update()

            pygame.draw.rect(window,green,[30,500,740,5])
            if reset_button[0]<= mx <= reset_button[0]+reset_button[2] and reset_button[1] <= my <= reset_button[1]+reset_button[3]:
                pygame.draw.rect(window, light_green,reset_button)
            else:
                pygame.draw.rect(window, green,reset_button)

            if predict_button[0]<= mx <= predict_button[0]+predict_button[2] and predict_button[1] <= my <= predict_button[1]+predict_button[3]:
                pygame.draw.rect(window, light_green, predict_button)
            else:
                pygame.draw.rect(window, green,predict_button)

            window.blit(predict_text,(predict_button[0]+8,predict_button[1]+2))
            window.blit(button_text,(reset_button[0]+15,reset_button[1]+2))
            pygame.display.update()
    
    except KeyboardInterrupt:
        print('')
        print('Exiting Demo...')

    
gameloop()           

pygame.quit()


