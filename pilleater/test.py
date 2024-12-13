import pygame
import sys

pygame.init()
screen = pygame.display.set_mode((640, 480))
pygame.display.set_caption('Pygame Test Window')

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    # Fill the screen with a color
    screen.fill((0, 128, 255))
    # Update the display
    pygame.display.flip()

pygame.quit()
sys.exit()
