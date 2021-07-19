from game import Game
import pygame



pygame.init()

display = pygame.display.set_mode((400, 800))
pygame.display.set_caption('Tetris')

clock = pygame.time.Clock()

game = Game()

locked = (0, 0, 255)
moving = (255, 0, 0)
white = (255, 255, 255)

def paint(buffer_):
    display.fill((0, 0, 0))
    for x in range(10):
        for y in range(20):
            if buffer_[x, y] == 0:
                pass
            elif buffer_[x, y] == 1:
                pygame.draw.rect(display, locked, pygame.Rect(x * 40, y * 40, 40, 40))
                pygame.draw.rect(display, white, pygame.Rect(x * 40, y * 40, 40, 40), 2)
            elif buffer_[x, y] == 2:
                pygame.draw.rect(display, moving, pygame.Rect(x * 40, y * 40, 40, 40))
                pygame.draw.rect(display, white, pygame.Rect(x * 40, y * 40, 40, 40), 2)
    pygame.display.update()

controls = {
    pygame.K_LEFT:'left',
    pygame.K_RIGHT:'right',
    pygame.K_SPACE:'rotate',
    pygame.K_DOWN:'drop'
    }

gameover = False

# exit()
while not gameover:
    cmd = 'nop'
    debug = False
    buffer = game.render()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            gameover = True
        if event.type == pygame.KEYDOWN and event.key in controls:
            cmd = controls[event.key]
        if event.type == pygame.KEYDOWN and event.key == pygame.K_b:
            print('Hello darkness my old friend')
            debug = True
        print(event)
    display.fill((0, 0, 0))
    for x in range(10):
        for y in range(20):
            if buffer[x, y] == 0:
                pass
            elif buffer[x, y] == 1:
                pygame.draw.rect(display, locked, pygame.Rect(x * 40, y * 40, 40, 40))
                pygame.draw.rect(display, white, pygame.Rect(x * 40, y * 40, 40, 40), 2)
            elif buffer[x, y] == 2:
                pygame.draw.rect(display, moving, pygame.Rect(x * 40, y * 40, 40, 40))
                pygame.draw.rect(display, white, pygame.Rect(x * 40, y * 40, 40, 40), 2)
    # cmd = 'nop'
    if debug:
        print('Thee')
        # pass
    game.step(cmd=cmd)
    pygame.display.update()
    clock.tick(5)
