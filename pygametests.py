# Based on DaFluffyPotato's MouseInput Pygame Tutorial
# https://www.youtube.com/watch?v=vhNiwvUv4Jw&ab_channel=DaFluffyPotato
# https://pastebin.com/7ndjJrM2
# Setup Python ----------------------------------------------- #
import pygame, sys
from datasets import *

with open('.//_Reports/TwoLayerTestNew/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

resize = transforms.Resize((32, 32))

# Setup pygame/window ---------------------------------------- #
mainClock = pygame.time.Clock()
from pygame.locals import *
pygame.init()
pygame.display.set_caption('game base')
screen = pygame.display.set_mode((640, 640), 0, 32)

# img = pygame.image.load('pic.png').convert()

offset = [0, 0]

clicking = False
right_clicking = False
middle_click = False
draw_set = set()

def distance(point1, point2):

    dx2 = (point1[0] - point2[0])**2
    dy2 = (point1[1] - point2[1])**2

    r = (dx2+dy2)**(0.5)

    return r


# Loop ------------------------------------------------------- #
while True:

    # Background --------------------------------------------- #
    screen.fill((255, 255, 255))

    mx, my = pygame.mouse.get_pos()
    radius = 30

    rot = 0
    loc = [mx, my]
    if clicking:
        draw_set.add((loc[0], loc[1]))
    if right_clicking:
        remove_set = set()
        for point in draw_set:
            if distance(loc, point) < radius+5:
                remove_set.add(point)
        draw_set = draw_set - remove_set
    if middle_click:
        pass
    # screen.blit(pygame.transform.rotate(img, rot), (loc[0] + offset[0], loc[1] + offset[1]))
    for point in draw_set:
        # pygame.draw.circle(screen, (0, 0, 0), (point[0], point[1]), radius)
        pygame.draw.rect(screen, (0, 0, 0),
                         pygame.Rect((point[0]-radius/2, point[1]-radius/2),
                                     (radius, radius)), width=0)

    # pygame.draw.circle(screen, (0, 0, 0), (loc[0], loc[1]), radius)
    pygame.draw.rect(screen, (0, 0, 0),
                     pygame.Rect((loc[0] - radius / 2, loc[1] - radius / 2),
                                 (radius, radius)), width=0)
    # Buttons ------------------------------------------------ #
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        if event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                pygame.quit()
                sys.exit()
        if event.type == MOUSEBUTTONDOWN:
            if event.button == 1:
                clicking = True
            if event.button == 3:
                right_clicking = True
            if event.button == 2:
                pass
            if event.button == 4:
                pass
            if event.button == 5:
                pass
        if event.type == MOUSEBUTTONUP:
            if event.button == 1:
                clicking = False
            if event.button == 3:
                right_clicking = False

    # Update ------------------------------------------------- #
    pygame.display.update()
    pygame.draw.circle(screen, (255, 255, 255), (loc[0], loc[1]), radius)
    pygame.image.save(screen, ".//_Data/pyimage.jpg")
    img = Image.open(".//_Data/pyimage.jpg")
    tsr = grayscale(img_to_tsr(resize(img))) - 0.5
    index = int(th.argmax(model(tsr.view(1, 1, 32, 32))))
    # plt.imshow(tsr.view(32, 32).detach())
    print(index, HASYv2Dataset.latex_dict[index])
    mainClock.tick(60)