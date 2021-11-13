# Based on DaFluffyPotato's MouseInput Pygame Tutorial
# https://www.youtube.com/watch?v=vhNiwvUv4Jw&ab_channel=DaFluffyPotato
# https://pastebin.com/7ndjJrM2
# Help with the active text box:
# https://stackoverflow.com/questions/46390231/how-can-i-create-a-text-input-box-with-pygame
# Setup Python ----------------------------------------------- #
import pygame, sys
from datasets import *
from pygame_classes import Button

print("Opening model...")
with open('.//_Reports/TwoLayerTest1 - 2021-11-12 16_34_06.229661/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

resize = transforms.Resize((32, 32))

# Setup pygame/window ---------------------------------------- #
mainClock = pygame.time.Clock()
from pygame.locals import *
print("Initializing Pygame...")
pygame.init()
font = pygame.font.Font(None, 32)   # what is this
# Drawing variables:
radius = 30
extra = 0.1 * radius

# Display layers:
pygame.display.set_caption('Drawing Board')
screen = pygame.display.set_mode((1280, 800), 0, 32)    # Base screen
drawing_board = pygame.Surface((640, 640))              # Drawing board
layer = pygame.Surface((640, 640))      # A layer on top for the mouse
layer.set_colorkey((255, 255, 255))     # Which is transparent to white colour
options = pygame.Surface((1280, 160))   # Options part (below)
text_box = pygame.Rect((240, 695), (600, 50), width=0)


# Button initialization:
button1 = Button('...', (810, 70), (300, 100), font=30, bg="gray")
button2 = Button('...', (810, 270), (300, 100), font=30, bg="gray")
button3 = Button('...', (810, 470), (300, 100), font=30, bg="gray")
button4 = Button('Clear', (70, 670), (100, 100), font=30, bg="white")

button_list = [
    button1, button2, button3, button4
]
# Runtime variables (related to mouse clicking):
clicking = False
right_clicking = False
middle_click = False
active = False
text = ''

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
    options.fill(("gray"))
    loc = pygame.mouse.get_pos()

    # loc = [mx, my]
    if clicking:
        draw_set.add(loc)
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
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        if event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                pygame.quit()
                sys.exit()
        if event.type == MOUSEBUTTONDOWN:
            for index, button in enumerate(button_list):
                if button.rect.collidepoint(*loc):
                    bool_val = button.click(event)
                    if bool_val:
                        draw_set = set()
                        if index < 3:
                            text += txt_list[index]

            if event.button == 1:
                clicking = True
                if text_box.collidepoint(*loc):
                    active = not active
                else:
                    active = False

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
        if event.type == KEYDOWN:
            if active:
                if event.key == K_RETURN:
                    print(text)
                    text = ''
                elif event.key == K_BACKSPACE:
                    text = text[:-1]
                else:
                    text += event.unicode

    # Update ------------------------------------------------- #
    screen.blit(drawing_board, (0,0))
    # pygame.draw.circle(screen, (0, 0, 0), (loc[0], loc[1]), radius)
    pygame.draw.rect(layer, (0, 0, 0),
                     pygame.Rect((loc[0] - radius / 2, loc[1] - radius / 2),
                                 (radius, radius)), width=0)
    screen.blit(layer, (0, 0))
    screen.blit(options, (0, 640))
    txt_surface = font.render(text, True, (255, 255, 255))
    width = max(600, txt_surface.get_width() + 10)
    text_box.w = width
    screen.blit(txt_surface, (text_box.x+5, text_box.y+5))
    pygame.draw.rect(screen, (255, 255, 255), text_box, 2)

    # Model --------------------------------------------------- #

    pygame.image.save(drawing_board, ".//_Data/pyimage.jpg")
    txt_list = []
    lbl_list = []
    val_list = []
    img = Image.open(".//_Data/pyimage.jpg")
    tsr = grayscale(img_to_tsr(resize(img))) - 0.5
    out = model(tsr.view(1, 1, 32, 32)).view(-1)
    out_copy = out.clone()
    for i in range(3):
        idx = int(th.argmax(out))
        label = HASYv2Dataset.latex_dict[idx]
        txt_list.append(label)
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

    for button in button_list:
        button.show(screen)

    button1.change_text(
        '{name}: {val:.4g}'.format(
            name=txt_list[0],val=val_list[0]))
    button2.change_text(
        '{name}: {val:.4g}'.format(
            name=txt_list[1], val=val_list[1]))
    button3.change_text(
        '{name}: {val:.4g}'.format(
            name=txt_list[2], val=val_list[2]))
    button4.change_text("Clear")

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