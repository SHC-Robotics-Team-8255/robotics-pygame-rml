import pygame
import time

pygame.init() # https://www.pygame.org/docs/

WINDOW_HEIGHT = 500
WINDOW_WIDTH = 800
FPS = 24

window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))

clock = pygame.time.Clock()


while True:

    window.fill((0, 0, 0))
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit(0)

    rectangle = pygame.Rect(0, 0, 200, 100)
    pygame.draw.rect(window, (0, 255, 0), rectangle)

    pygame.draw.circle(window, (255, 255, 0), (100, 400), 80)

    image = pygame.image.load("hi.png")
    image_boundaries = image.get_rect()
    image_boundaries.move_ip(200, 0)
    window.blit(image, image_boundaries)

    clock.tick(FPS)
    pygame.display.flip()

time.sleep(5)
pygame.quit()