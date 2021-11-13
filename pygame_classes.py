# https://pythonprogramming.altervista.org/buttons-in-pygame/
import pygame
pygame.font.init()

class Button:
    """Create a button, then blit the surface in the while loop"""

    def __init__(self, text, pos, size, font, bg="black", feedback=""):
        self.x, self.y = pos
        self.size = size
        self.font = pygame.font.SysFont("timesnewroman", font)
        if feedback == "":
            self.feedback = text
        else:
            self.feedback = feedback
        self.bg = bg
        self.change_text(text, bg)

    def change_text(self, text: str, bg: str = None):
        """Change the text whe you click"""
        if bg is None:
            bg = self.bg
        self.text = self.font.render(text, 1, pygame.Color("black"))
        self.surface = pygame.Surface(self.size)
        self.text_size = self.text.get_size()
        self.surface.fill(bg)
        self.surface.blit(self.text, (
            (self.size[0] - self.text_size[0]) / 2,
            (self.size[1] - self.text_size[1]) / 2
        ))
        self.rect = pygame.Rect(self.x, self.y, self.size[0], self.size[1])

    # def show(self, screen: pygame.display, button: 'Button'):
    def show(self, screen: pygame.display):
        screen.blit(self.surface, (self.x, self.y))

    def click(self, event):
        x, y = pygame.mouse.get_pos()
        if event.type == pygame.MOUSEBUTTONDOWN:
            if pygame.mouse.get_pressed()[0]:
                if self.rect.collidepoint(x, y):
                    self.change_text(self.feedback, bg="red")
                    return True

        return False
