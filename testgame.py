from game import Game, INTS
import pygame



pygame.init()

display = pygame.display.set_mode((400, 800))
pygame.display.set_caption('Tetris')

clock = pygame.time.Clock()

game = Game()

locked = (128, 128, 128)
moving = {
    'I':(0,255,255),
    'J':(0,0,255),
    'L':(255, 165, 0),
    'O':(255,255,0),
    'S':(0,255,0),
    'T':(128,0,128),
    'Z':(255,0,0)
}

TOINTS = {v: k for k, v in INTS.items()}

white = (255, 255, 255)

def paint(buffer_):
    display.fill((0, 0, 0))
    for x in range(10):
        for y in range(20):
            if buffer_[x, y] == 0:
                pass
            elif buffer[x, y] in range(1,8):
                pygame.draw.rect(display, moving[TOINTS[buffer_[x, y]]], pygame.Rect(x * 40, y * 40, 40, 40))
                pygame.draw.rect(display, white, pygame.Rect(x * 40, y * 40, 40, 40), 2)
            elif buffer[x, y] == 8:
                pygame.draw.rect(display, moving[game.current.type], pygame.Rect(x * 40, y * 40, 40, 40))
                pygame.draw.rect(display, white, pygame.Rect(x * 40, y * 40, 40, 40), 2)
    pygame.display.update()

controls = {
    pygame.K_LEFT:'left',
    pygame.K_RIGHT:'right',
    pygame.K_SPACE:'rotate',
    pygame.K_DOWN:'drop',
    pygame.K_h:'hold'
    }

# gameover = False

debug = False
rotatedebug = False
paused = False

game.timetick()
# exit()
while not game.gameover:
    cmd = 'nop'

    buffer = game.render()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game.gameover = True
        if event.type == pygame.KEYDOWN and event.key in controls:
            cmd = controls[event.key]
            # game.step(cmd)
        if event.type == pygame.KEYDOWN and event.key == pygame.K_b:
            print('Hello darkness my old friend')
            debug = True
        if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
            print('ROTATO')
            rotatedebug = not rotatedebug
        if event.type == pygame.KEYDOWN and event.key == pygame.K_p:
            paused = not paused
        # print(event)
    paint(game.render())
    if debug:
        print('Thee')
        # pass
    if rotatedebug:
        cmd='rotate'
    game.step(cmd=cmd)
    pygame.display.update()
    # clock.tick(5)
