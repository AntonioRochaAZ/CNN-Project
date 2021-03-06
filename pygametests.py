# Based on DaFluffyPotato's MouseInput Pygame Tutorial
# https://www.youtube.com/watch?v=vhNiwvUv4Jw&ab_channel=DaFluffyPotato
# https://pastebin.com/7ndjJrM2
# Help with the active text box:
# https://stackoverflow.com/questions/46390231/how-can-i-create-a-text-input-box-with-pygame
# Setup Python ----------------------------------------------- #
import pygame, sys
from pygame.locals import *
from pygame_classes import Button
from datasets import *


# Setup Model ------------------------------------------------ #
print("Opening model...")
with open(
        './/_Reports/TwoLayerTest1 - 2021-11-12 16_34_06.229661/best_model.pkl',
        'rb') as f:
    model = pickle.load(f)

resize = transforms.Resize((32, 32))
txt_list = []
lbl_list = []
val_list = []

# Setup pygame/window ---------------------------------------- #
print("Initializing Pygame...")

# Initialization:
mainClock = pygame.time.Clock()
pygame.init()
font = pygame.font.Font(None, 32)   # what is this

# Display layers:
pygame.display.set_caption('Drawing Board')
screen = pygame.display.set_mode((1280, 800), 0, 32)    # Base screen
drawing_board = pygame.Surface((640, 640))              # Drawing board
layer = pygame.Surface((640, 640))      # A layer on top for the mouse
layer.set_colorkey((255, 255, 255))     # Which is transparent to white colour
options = pygame.Surface((1280, 160))   # Options part (below)
text_box = pygame.Rect((240, 695), (600, 50), width=0)  # Text box

# Button initialization:
button1 = Button('...', (810, 70), (300, 100), font=30, bg="gray")
button2 = Button('...', (810, 270), (300, 100), font=30, bg="gray")
button3 = Button('...', (810, 470), (300, 100), font=30, bg="gray")
button4 = Button('Clear', (70, 670), (100, 100), font=30, bg="white")

button_list = [
    button1, button2, button3, button4
]

# Runtime variables (related to mouse clicking and text box):
clicking = False
right_clicking = False
middle_click = False
active = False
text = ''

# Drawing variables:
side = 30               # Square's side length
extra = 0.1 * side      # Extra range for erasing points
draw_set = set()        # Set of points to be drawn

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
    options.fill("gray")

    # Mouse handling ----------------------------------------- #
    loc = pygame.mouse.get_pos()

    if clicking:
        draw_set.add(loc)
    if right_clicking:
        remove_set = set()
        for point in draw_set:  # erase points next to the mouse.
            if distance(loc, point) < side+extra:
                remove_set.add(point)
        draw_set = draw_set - remove_set
    # if middle_click:            # No implementation for the moment.
    #     pass

    # Drawing the points ------------------------------------- #
    for point in draw_set:      # Drawing the points.
        pygame.draw.rect(drawing_board, (0, 0, 0),
                         pygame.Rect((point[0] - side / 2, point[1] - side / 2),
                                     (side, side)), width=0)

    # Events ------------------------------------------------- #
    for event in pygame.event.get():

        # Quitting:
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        if event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                pygame.quit()
                sys.exit()

        # Pressing mouse buttons:
        if event.type == MOUSEBUTTONDOWN:

            if event.button == 1:
                clicking = True
                # Checking button presses:
                for index, button in enumerate(button_list):
                    if button.rect.collidepoint(*loc):
                        button.click(event)
                        draw_set = set()  # Clear the drawing
                        if index < 3:  # If it's not the clear button
                            text += txt_list[index]  # Add to the text box

                if text_box.collidepoint(*loc):
                    active = not active
                else:
                    active = False

            if event.button == 3:
                right_clicking = True

            # Not implemented:
            # if event.button == 2:
            #     pass
            # if event.button == 4:
            #     pass
            # if event.button == 5:
            #     pass

        # Releasing mouse buttons:
        if event.type == MOUSEBUTTONUP:
            if event.button == 1:
                clicking = False
            if event.button == 3:
                right_clicking = False

        # Typing:
        if event.type == KEYDOWN:
            # If the text box is active:
            if active:
                # Not implemented: pressing return
                # if event.key == K_RETURN:
                #     print(text)
                #     text = ''
                #     pass
                if event.key == K_BACKSPACE:
                    text = text[:-1]
                else:
                    text += event.unicode

    # Update ------------------------------------------------- #
    screen.blit(drawing_board, (0,0))
    pygame.draw.rect(layer, (0, 0, 0),
                     pygame.Rect((loc[0] - side / 2, loc[1] - side / 2),
                                 (side, side)), width=0)    # Mouse arrow
    screen.blit(layer, (0, 0))
    screen.blit(options, (0, 640))

    # Text box:
    txt_surface = font.render(text, True, (255, 255, 255))
    width = max(600, txt_surface.get_width() + 10)
    text_box.w = width
    screen.blit(txt_surface, (text_box.x+5, text_box.y+5))
    pygame.draw.rect(screen, (255, 255, 255), text_box, 2)

    # Model Prediction --------------------------------------- #
    # TODO:
    #   Add the possibility of only making predictions every couple of loops.
    pygame.image.save(drawing_board, ".//_Data/pyimage.jpg")
    txt_list = []
    lbl_list = []
    val_list = []
    img = Image.open(".//_Data/pyimage.jpg")
    tsr = grayscale(img_to_tsr(resize(img))) - 0.5
    out = model(tsr.view(1, 1, 32, 32)).view(-1)
    out_copy = out.clone()

    # Taking the best three predictions:
    for i in range(3):
        idx = int(th.argmax(out))               # Index
        label = HASYv2Dataset.latex_dict[idx]
        txt_list.append(label)                  # Latex command
        if label.islower():
            label = "_"+label
        elif label == '/':
            label = "_slash"
        lbl_list.append(label)                  # _Assets image name
        val_list.append(out_copy[idx])          # "Probability"
        out[idx] = 0


    # TODO:
    #   Improve the following code to avoid repetition.
    try:
        img1 = pygame.image.load(f'.//_Assets/{lbl_list[0]}.png').convert()
        img1 = pygame.transform.scale(img1, (160, 144))
        img1.set_colorkey((255, 255, 255))
        screen.blit(img1, (1120, 50))
    except FileNotFoundError:       # Because not all Assets are ready.
        pass
    try:
        img2 = pygame.image.load(f'.//_Assets/{lbl_list[1]}.png').convert()
        img2 = pygame.transform.scale(img2, (160, 144))
        img2.set_colorkey((255, 255, 255))
        screen.blit(img2, (1120, 250))
    except FileNotFoundError:       # Because not all Assets are ready.
        pass
    try:
        img3 = pygame.image.load(f'.//_Assets/{lbl_list[2]}.png').convert()
        img3 = pygame.transform.scale(img3, (160, 144))
        img3.set_colorkey((255, 255, 255))
        screen.blit(img3, (1120, 450))
    except FileNotFoundError:       # Because not all Assets are ready.
        pass

    # Show the buttons (for some reason this has to come before changing text)
    for button in button_list:
        button.show(screen)

    # Changing their text
    button1.change_text(
        '{name}: {val:.4g}'.format(
            name=txt_list[0], val=val_list[0]))
    button2.change_text(
        '{name}: {val:.4g}'.format(
            name=txt_list[1], val=val_list[1]))
    button3.change_text(
        '{name}: {val:.4g}'.format(
            name=txt_list[2], val=val_list[2]))
    button4.change_text("Clear")

    pygame.display.update()
    mainClock.tick(60)

