# Based on DaFluffyPotato's MouseInput Pygame Tutorial
# https://www.youtube.com/watch?v=vhNiwvUv4Jw&ab_channel=DaFluffyPotato
# https://pastebin.com/7ndjJrM2
# Setup Python ----------------------------------------------- #
import pygame, sys
from datasets import *
from pygame_classes import Button

with open('.//_Reports/TwoLayerTest100 - 2021-11-11 16_15_22.806234/small_model.pkl', 'rb') as f:
    model = pickle.load(f)

resize = transforms.Resize((32, 32))

# Setup pygame/window ---------------------------------------- #
mainClock = pygame.time.Clock()
from pygame.locals import *
pygame.init()

# Drawing variables:
radius = 30
extra = 0.1 * radius

# Display layers:
pygame.display.set_caption('Drawing Board')
screen = pygame.display.set_mode((1280, 640), 0, 32)    # Base screen
drawing_board = pygame.Surface((640, 640))              # Drawing board
layer = pygame.Surface((640, 640))      # A layer on top for the mouse
layer.set_colorkey((255, 255, 255))     # Which is transparent to white colour

# Button initialization:
button1 = Button('...', (810, 70), (300, 100), font=30, bg="gray", feedback="")
button2 = Button('...', (810, 270), (300, 100), font=30, bg="gray", feedback="")
button3 = Button('...', (810, 470), (300, 100), font=30, bg="gray", feedback="")

# Runtime variables (related to mouse clicking):
clicking = False
right_clicking = False
middle_click = False

# Set of points to be drawn:
draw_set = set()

# Function for calculating the distance between two points (for erasing).
def distance(point1, point2):
    dx2 = (point1[0] - point2[0])**2
    dy2 = (point1[1] - point2[1])**2
    r = (dx2+dy2)**(0.5)
    return r

# Loop ------------------------------------------------------- #
while True:

    # Background --------------------------------------------- #
    screen.fill((116, 125, 207))
    drawing_board.fill((255, 255, 255))
    layer.fill((255, 255, 255))

    loc = pygame.mouse.get_pos()

    # loc = [mx, my]
    if clicking:
        draw_set.add((loc[0], loc[1]))
    if right_clicking:
        remove_set = set()
        for point in draw_set:
            if distance(loc, point) < radius+extra:
                remove_set.add(point)
        draw_set = draw_set - remove_set
    if middle_click:
        pass
    for point in draw_set:
        # pygame.draw.circle(screen, (0, 0, 0), (point[0], point[1]), radius)
        pygame.draw.rect(drawing_board, (0, 0, 0),
                         pygame.Rect((point[0]-radius/2, point[1]-radius/2),
                                     (radius, radius)), width=0)
    # Buttons ------------------------------------------------ #
    for event in pygame.event.get():
        button1.click(event)
        button2.click(event)
        button3.click(event)
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
    button1.show(screen, button1)
    button2.show(screen, button2)
    button3.show(screen, button3)

    # Update ------------------------------------------------- #
    screen.blit(drawing_board, (0,0))
    # pygame.draw.circle(screen, (0, 0, 0), (loc[0], loc[1]), radius)
    pygame.draw.rect(layer, (0, 0, 0),
                     pygame.Rect((loc[0] - radius / 2, loc[1] - radius / 2),
                                 (radius, radius)), width=0)
    screen.blit(layer, (0, 0))

    # Model --------------------------------------------------- #

    pygame.image.save(drawing_board, ".//_Data/pyimage.jpg")

    lbl_list = []
    val_list = []
    img = Image.open(".//_Data/pyimage.jpg")
    tsr = grayscale(img_to_tsr(resize(img))) - 0.5
    out = model(tsr.view(1, 1, 32, 32)).view(-1)
    out_copy = out.clone()
    for i in range(3):
        idx = int(th.argmax(out))
        label = HASYv2Dataset.latex_dict[idx]
        if label.islower():
            label = "_"+label
        lbl_list.append(label)
        val_list.append(out_copy[idx])
        out[idx] = 0
    #
    #
    # index_2 = int(th.argmax(out))
    # val_2 = out_copy[index_2]
    # out[index_2] = 0
    # index_3 = int(th.argmax(out))
    # val_3 = out_copy[index_3]
    # Images (will be updated)

    try:
        img1 = pygame.image.load(f'.//_Assets/{lbl_list[0]}.png').convert()
        img1 = pygame.transform.scale(img1, (160, 144))
        img1.set_colorkey((255, 255, 255))
        screen.blit(img1, (1120, 50))
    except FileNotFoundError:
        pass
    try:
        img2 = pygame.image.load(f'.//_Assets/{lbl_list[1]}.png').convert()
        img2 = pygame.transform.scale(img2, (160, 144))
        img2.set_colorkey((255, 255, 255))
        screen.blit(img2, (1120, 250))
    except FileNotFoundError:
        pass
    try:
        img3 = pygame.image.load(f'.//_Assets/{lbl_list[2]}.png').convert()
        img3 = pygame.transform.scale(img3, (160, 144))
        img3.set_colorkey((255, 255, 255))
        screen.blit(img3, (1120, 450))
    except FileNotFoundError:
        pass

    button1.change_text(
        '{name}: {val:.4g}'.format(
            name=lbl_list[0],val=val_list[0]))
    button2.change_text(
        '{name}: {val:.4g}'.format(
            name=lbl_list[1], val=val_list[1]))
    button3.change_text(
        '{name}: {val:.4g}'.format(
            name=lbl_list[2], val=val_list[2]))
    # button3 = Button(
    #     '{name}: {val:.4g}'.format(name=HASYv2Dataset.latex_dict[index_3], val=val_3),
    #     (940, 600), (300, 100),
    #     font=30,
    #     bg="gray",
    #     feedback="You clicked me")

    # plt.imshow(tsr.view(32, 32).detach())
    # print(index, HASYv2Dataset.latex_dict[index])

    pygame.display.update()
    mainClock.tick(60)