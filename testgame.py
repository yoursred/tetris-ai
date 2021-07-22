from game import Game
import pygame

game = Game(render=True)

controls = {
    pygame.K_LEFT:'left',
    pygame.K_RIGHT:'right',
    pygame.K_SPACE:'rotate',
    pygame.K_DOWN:'drop',
    pygame.K_h:'hold'
    }

paused = False
game.tickdelay = 1
game.timetick()

hold = False

while not game.gameover:
    cmd = 'nop'

    buffer = game.render_board()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game.gameover = True
        if event.type == pygame.KEYDOWN and event.key in controls:
            cmd = controls[event.key]
        if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
            print('HOLD')
            hold = not hold
        if event.type == pygame.KEYDOWN and event.key == pygame.K_p:
            paused = not paused

    if paused:
        game.timer.cancel()
        game.ticking = False
    if not paused and not game.ticking:
        game.timetick()
    if hold:
        game.game_step('hold')

    game.game_step(cmd=cmd)
    game.paint(cmd)
    pygame.display.update()

print(game.score)