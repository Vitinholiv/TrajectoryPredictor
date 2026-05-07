# Classe do simulador visual

import pygame
import torch
import numpy as np
from src.core.predictor import TrajectoryPredictor

class Simulator:
    """
    Classe responsável pela interface gráfica interativa e simulação em tempo real.
    
    Utiliza Pygame para renderizar um ambiente onde o usuário pode tentar acertar 
    alvos manualmente ou utilizando as predições da Rede Neural. Gerencia estados 
    de jogo, entrada de usuário e visualização de entidades físicas.

    Args:
        predictor (TrajectoryPredictor): Instância do preditor contendo o modelo 
            treinado para auxílio de mira.
    """

    def __init__(self, predictor: TrajectoryPredictor):
        self.p = predictor

    def simulate(self, width=1280, height=720, fps=60, adaptativeScreen=True):
        """
        Inicia a simulação gráfica.

        Args:
            width (int): Largura da janela em pixels.
            height (int): Altura da janela em pixels.
            fps (int): Taxa de quadros por segundo da simulação.
            adaptativeScreen (bool): Se True, a câmera ajusta o zoom e posição 
                automaticamente para manter os elementos visíveis.

        Returns:
            None: Finaliza a execução ao fechar a janela do Pygame.
        """

        pygame.init()
        screen = pygame.display.set_mode((width, height))
        font = pygame.font.SysFont("Arial", 24)
        pygame.display.set_caption("Trajectory Predictor Simulator")
        clock = pygame.time.Clock()

        STATE_AIM = 0; STATE_SIMULATE = 1; STATE_LOSE = 2; STATE_WIN = 3
        PORTAL = 0; PLANET = 1; PROJ = 2; PIECE = 3
        PORTAL_RPS = 1; PLANET_RPS = 10; PROJECTILE_RPS = 5
        AI_ARROW_COLOR = (255, 215, 0); ARROW_COLOR = (200, 220, 255)
        DRAG_MIN = 1; DRAG_MAX = 150; HIT_RADIUS = 1
        WIN_TIME = 1; LOSE_TIME = 1
        
        ENTRY_IMG = pygame.image.load('res/portal.png')
        TARGET_IMG = pygame.image.load('res/redPortal.png')
        PLANET_IMG = pygame.image.load('res/singularity.png')
        PROJECTILE_IMG = pygame.image.load('res/projectile.png')
        BROKEN_PROJECTILE_IMG = pygame.image.load('res/brokenProjectile.png')

        projectile = [.0,.0,.0,.0] 
        target = [.0,.0,.0]; entry = [.0,.0,.0] 
        planets = [[.0,.0,.0,.0]] 
        minX = 0; minY = 0; maxX = 1; maxY = 1
        lSpeed = 0; lAngle = 0; predSpeed = 0; predAngle = 0
        arrow = [.0, .0]; predArrow = [.0, .0]; projAngle = .0
        screenZero = [.0,.0,.0,.0]
        pieces = []; winDist = 0; winDir = 1; winAngDelta = 0

        running = True; gameState = STATE_AIM; dragging = False
        startTime = 0; score = 0

        def to_screen(x, y):
            nonlocal minX,minY,maxX,maxY
            return int(((x-minX)/(maxX-minX))*width),int((1-((y-minY)/(maxY-minY)))*height)
        
        def to_screen_len(k):
            nonlocal minX,minY,maxX,maxY
            return k*width/(maxX-minX)
        
        def update_screen_borders():
            nonlocal minX,minY,maxX,maxY,planets,projectile,target
            pX = [planets[i][0] for i in range(len(planets))]
            pY = [planets[i][1] for i in range(len(planets))]

            minX = min(0,projectile[0],target[0],*pX)
            minY = min(0,projectile[1],target[1],*pY)
            maxX = max(0,projectile[0],target[0],*pX)
            maxY = max(0,projectile[1],target[1],*pY)

            marginX = (maxX - minX)*0.125; marginY = (maxY - minY)*0.125
            minX -= marginX; maxX += marginX
            minY -= marginY; maxY += marginY

            if (maxX-minX)/(maxY-minY) < 16/9:
                difr = (maxY-minY)*16/9 - (maxX-minX)
                minX -= difr/2; maxX += difr/2
            elif (maxX-minX)/(maxY-minY) > 16/9:
                difr = (maxX-minX)*9/16 - (maxY-minY)
                minY -= difr/2; maxY += difr/2

        def reset_game_state():
            nonlocal projectile, target, planets, minX, minY, maxX, maxY, predArrow, predSpeed, predAngle, screenZero, pieces, score, gameState
            pieces = []; projectile = [.0,.0,.0,.0]
            update_screen_borders()
            gameState = STATE_AIM

        def new_game_state():
            nonlocal projectile, target, planets, minX, minY, maxX, maxY, predArrow, predSpeed, predAngle, screenZero, pieces, score, gameState
            pieces = []; projectile = [.0,.0,.0,.0]
            
            if self.p.hp['architecture'] == 'SingleAim':
                planets = [[float(self.p.bX[i]),float(self.p.bY[i]),float(self.p.bMass[i]),.0] for i in range(len(self.p.bX))]
            
            _alpha = np.random.rand() * (2 * np.pi)
            _r = np.sqrt(np.random.rand()) * self.p.hp['maxDistance']
            _indx = np.random.choice([i for i, body in enumerate(self.p.hp['bodies']) if body.get('habitable', True) is True])
            target = [_r * np.cos(_alpha) + planets[_indx][0], _r * np.sin(_alpha) + planets[_indx][1], .0]

            update_screen_borders()
            cX,cY = to_screen(0,0)
            screenZero = [minX,maxX,minY,maxY]

            if self.p.hp['architecture'] == 'SingleAim':
                tensorPos = torch.tensor(target[:2],device=self.p.hp['device'],dtype=torch.float32).view(1,2)
                predSpeed, predAngle, _ = self.p.architecture(tensorPos)
                predSpeed = float(predSpeed); predAngle = float(predAngle)

                predProgress = (predSpeed - self.p.hp['limSpeed'][0])/(self.p.hp['limSpeed'][1] - self.p.hp['limSpeed'][0])
                predArrow = [int(cX + predProgress * DRAG_MAX * np.cos(predAngle)), int(cY - DRAG_MAX * predProgress * np.sin(predAngle))]
            
            gameState = STATE_AIM
        
        def handle_events():
            nonlocal running, gameState, startTime, dragging, lSpeed, lAngle, predSpeed, predAngle, projectile, score
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pygame.quit()
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and gameState == STATE_AIM:
                    dragging = True
                elif (event.type == pygame.MOUSEBUTTONUP and event.button == 1 and dragging) or (event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE and gameState == STATE_AIM):
                    dragging = False
                    gameState = STATE_SIMULATE
                    startTime = pygame.time.get_ticks()
                    if event.type != pygame.MOUSEBUTTONUP:
                        lSpeed,lAngle = predSpeed,predAngle
                    projectile = [.0,.0,lSpeed * np.cos(lAngle),lSpeed * np.sin(lAngle)]
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE and gameState == STATE_SIMULATE:
                    score -= 1; gameState = STATE_AIM; reset_game_state()
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE and gameState == STATE_AIM:
                    new_game_state()

        def game_logic():
            nonlocal clock, entry, target, dragging, lAngle, lSpeed, predAngle, predSpeed, arrow, predArrow, gameState, projectile, score, startTime
            nonlocal minX, minY, maxX, maxY, projAngle, pieces, winDist, winDir, winAngDelta
            dt = clock.get_time()/1000

            dRad = [PORTAL_RPS * dt * 360, PLANET_RPS * dt * 360, PROJECTILE_RPS * dt * 360]
            entry[2] = (entry[2] + dRad[PORTAL]) % 360
            target[2] = (target[2] + dRad[PORTAL]) % 360
            projAngle = (projAngle + dRad[PROJ]) % 360
    
            for p in planets:
                p[3] = (p[3] + dRad[PLANET]) % 360

            if dragging:
                cX,cY = to_screen(0,0)
                mousePos = pygame.mouse.get_pos()
                dxScreen,dyScreen = mousePos[0] - cX,mousePos[1] - cY
                lAngle = np.atan2(-dyScreen,dxScreen)
                dragDist = max(DRAG_MIN, min(np.sqrt(dxScreen**2 + dyScreen**2), DRAG_MAX))
                speedPercent = (dragDist - DRAG_MIN)/(DRAG_MAX - DRAG_MIN)
                lSpeed = self.p.hp['limSpeed'][0] + speedPercent * (self.p.hp['limSpeed'][1] - self.p.hp['limSpeed'][0])
                arrow = [int(cX + dragDist * np.cos(lAngle)), int(cY - dragDist * np.sin(lAngle))]
            elif gameState == STATE_SIMULATE:
                projectileT = [torch.Tensor([i]) for i in projectile]
                nx, ny, nvx, nvy = self.p.lossFuncStep(self.p.bX, self.p.bY, self.p.bMass, projectileT[0], projectileT[1], projectileT[2], projectileT[3], self.p.hp['bias'], torch.Tensor([dt]))
                projectile = [nx.item(), ny.item(), nvx.item(), nvy.item()]

                if adaptativeScreen: update_screen_borders()
                
                if np.sqrt((projectile[0] - target[0])**2 + (projectile[1] - target[1])**2) < HIT_RADIUS:
                    score += 1; startTime = pygame.time.get_ticks()
                    winDist = np.sqrt((projectile[0] - target[0])**2 + (projectile[1] - target[1])**2)
                    winDx = (projectile[0] - target[0])/winDist; winDy = (projectile[1] - target[1])/winDist
                    winRefAng = np.atan2(winDy,winDx)
                    dotProd = (winDy * projectile[2]) + (- winDx * projectile[3])
                    winDir = 1 if dotProd < 0 else -1
                    winAngDelta = winRefAng - winDir*target[2]*(np.pi/180.0)
                    gameState = STATE_WIN
                elif (pygame.time.get_ticks() - startTime) > self.p.hp['limTime'][1]*1000:
                    score -= 1
                    for i in range(5):
                        ang = (torch.rand(1) * (2*torch.pi)).item()
                        spd = (self.p.hp['limSpeed'][0] + self.p.hp['limSpeed'][1])/2.0
                        pieces.append([projectile[0],projectile[1],spd*np.cos(ang),spd*np.sin(ang)])
                    startTime = pygame.time.get_ticks()
                    gameState = STATE_LOSE
            elif gameState == STATE_LOSE:
                for i_pc in range(len(pieces)):
                    pc = pieces[i_pc]
                    nx, ny, nvx, nvy = self.p.lossFuncStep(self.p.bX, self.p.bY, self.p.bMass, pc[0], pc[1], pc[2], pc[3], self.p.hp['bias'], torch.Tensor([dt]))
                    pieces[i_pc] = [nx.item(), ny.item(), nvx.item(), nvy.item()]
                if adaptativeScreen: update_screen_borders()
                if (pygame.time.get_ticks() - startTime) > (LOSE_TIME)*1000: reset_game_state()
            elif gameState == STATE_WIN:
                cDist = max(0, winDist*(1 - (pygame.time.get_ticks() - startTime)/(WIN_TIME*1000)))
                trueRefAng = winDir*target[2]*(np.pi/180.0) + winAngDelta
                projectile = [np.cos(trueRefAng)*cDist + target[0], np.sin(trueRefAng)*cDist + target[1], 0.0, 0.0]
                if (pygame.time.get_ticks() - startTime) > (WIN_TIME)*1000: new_game_state()
        
        def draw():
            nonlocal minX,minY,maxX,maxY,screenZero,projectile,planets,target,entry,projAngle,pieces
            screen.fill((0,0,0)); niceUnit = to_screen_len((screenZero[1]-screenZero[0])/50)
            for p in planets:
                px, py, mass, angle = p
                sx, sy = to_screen(px, py)
                scaled = pygame.transform.scale(PLANET_IMG,(int(5*niceUnit),int(5*niceUnit)))
                rotated = pygame.transform.rotate(scaled,angle)
                screen.blit(rotated, rotated.get_rect(center=(int(sx), int(sy))))

            cX,cY = to_screen(0,0); tX,tY = to_screen(target[0],target[1])
            prX,prY = to_screen(projectile[0],projectile[1])

            scaled = pygame.transform.scale(ENTRY_IMG,(max(int(1.5*niceUnit),1),max(int(1.5*niceUnit),1)))
            rotated = pygame.transform.rotate(scaled,entry[2])
            screen.blit(rotated, rotated.get_rect(center=(cX,cY)))

            scaled = pygame.transform.scale(TARGET_IMG,(max(1,int(1.5*niceUnit)),max(int(1.5*niceUnit),1)))
            rotated = pygame.transform.rotate(scaled,target[2])
            screen.blit(rotated, rotated.get_rect(center=(tX,tY)))

            if gameState == STATE_AIM:
                if predArrow[0] != 0 or predArrow[1] != 0:
                    pygame.draw.line(screen, AI_ARROW_COLOR, (cX, cY), predArrow, 2)
                    pygame.draw.circle(screen, (255, 215, 0), predArrow, 4)
                if dragging:
                    pygame.draw.line(screen, ARROW_COLOR, (cX, cY), arrow, 2)
                    pygame.draw.circle(screen, (255, 255, 255), arrow, 4)
                ctrl_text = "Space to use AI trajectory, Mouse to control"
                ctrl_surf = font.render(ctrl_text, True, (255, 255, 255))
                ctrl_surf.set_alpha(70)
                screen.blit(ctrl_surf, (width - ctrl_surf.get_width() - 20, height - 40))

            elif gameState == STATE_SIMULATE or gameState == STATE_WIN:
                scaled = pygame.transform.scale(PROJECTILE_IMG,(max(1,int(niceUnit)),max(1,int(niceUnit))))
                rotated = pygame.transform.rotate(scaled,projAngle)
                screen.blit(rotated, rotated.get_rect(center=(prX,prY)))

            elif gameState == STATE_LOSE:
                for i in range(len(pieces)):
                    scaled = pygame.transform.scale(BROKEN_PROJECTILE_IMG,(max(1,int(0.5*niceUnit)),max(1,int(0.5*niceUnit))))
                    rotated = pygame.transform.rotate(scaled,projAngle)
                    screen.blit(rotated, rotated.get_rect(center=(to_screen(pieces[i][0],pieces[i][1]))))

            score_surf = font.render(f"Score: {score}", True, (255, 255, 255))
            score_surf.set_alpha(150)
            screen.blit(score_surf, (20, height - 40))

            pygame.display.flip()

        new_game_state()
        running = True
        while running:
            handle_events()
            if not running: break
            game_logic()
            draw()
            clock.tick(fps)