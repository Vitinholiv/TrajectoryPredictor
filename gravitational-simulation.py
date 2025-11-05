import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import time
import math

# Valores Padrão dos Hiperparâmetros - Modelo de Simulação
DEF_nodes = 96
DEF_minSpeed = 1
DEF_maxSpeed = 20
DEF_minTime = 1
DEF_maxTime = 10
DEF_learningRate = 1e-4
DEF_pointsPerLoss = 1024
DEF_maxRadius = 20
DEF_xBody = 20
DEF_yBody = 20
DEF_attraction = 7500.0
DEF_divisionsPerSimulation = 200
DEF_epochs = 100000
DEF_angle = torch.pi*2

simulated_learner = {
    'identifier': 'simulated_learner',
    'nodes': DEF_nodes,
    'minSpeed': DEF_minSpeed,
    'maxSpeed': DEF_maxSpeed,
    'minTime': DEF_minTime,
    'maxTime': DEF_maxTime,
    'learningRate': DEF_learningRate,
    'pointsPerLoss': DEF_pointsPerLoss,
    'maxRadius': DEF_maxRadius,
    'xBody': DEF_xBody,
    'yBody': DEF_yBody,
    'attraction': DEF_attraction,
    'divisionsPerSimulation': DEF_divisionsPerSimulation,
    'epochs': DEF_epochs,
    'angle': DEF_angle
}

quick_learner = {
    'identifier': 'quick_learner',
    'nodes': DEF_nodes,
    'minSpeed': DEF_minSpeed,
    'maxSpeed': DEF_maxSpeed,
    'minTime': DEF_minTime,
    'maxTime': DEF_maxTime,
    'learningRate': DEF_learningRate,
    'pointsPerLoss': DEF_pointsPerLoss,
    'maxRadius': DEF_maxRadius,
    'xBody': DEF_xBody,
    'yBody': DEF_yBody,
    'attraction': DEF_attraction,
    'divisionsPerSimulation': DEF_divisionsPerSimulation,
    'epochs': 10000,
    'angle': DEF_angle
}

selected_model = simulated_learner

class TripleReluModel(nn.Module):
    def __init__(self, nodes, minSpeed, maxSpeed, minTime, maxTime, angle):
        super().__init__()
        self.minSpeed = minSpeed
        self.maxSpeed = maxSpeed
        self.minTime = minTime
        self.maxTime = maxTime
        self.angle = angle
        self.network = nn.Sequential(
            nn.Linear(2, nodes),
            nn.ReLU(),
            nn.Linear(nodes, nodes),
            nn.ReLU(),
            nn.Linear(nodes, nodes),
            nn.ReLU(),
            nn.Linear(nodes, 3)
        )
        
    def forward(self, xy):
        params = self.network(xy)
        v = self.minSpeed
        if self.maxSpeed > 0:
            v += nn.functional.sigmoid(params[:, 0:1])*(self.maxSpeed - self.minSpeed)
        else:
            v += nn.functional.softplus(params[:, 0:1])
        theta = torch.sigmoid(params[:, 1:2]) * self.angle
        t = self.minTime + torch.sigmoid(params[:, 2:3])*(self.maxTime - self.minTime)
        return v, theta, t
    
class TrajectoryPredictor:
    def __init__(self, identifier, load = False, nodes = DEF_nodes, minSpeed = DEF_minSpeed, maxSpeed = DEF_maxSpeed, minTime = DEF_minTime, maxTime = DEF_maxTime, learningRate = DEF_learningRate, pointsPerLoss = DEF_pointsPerLoss, maxRadius = DEF_maxRadius, xBody = DEF_xBody, yBody = DEF_yBody, attraction = DEF_attraction, divisionsPerSimulation = DEF_divisionsPerSimulation, epochs = DEF_epochs, angle = DEF_angle):
        
        self.identifier = identifier
        self.nodes = nodes
        self.minSpeed = minSpeed
        self.maxSpeed = maxSpeed
        self.minTime = minTime
        self.maxTime = maxTime
        self.learningRate = learningRate
        self.pointsPerLoss = pointsPerLoss
        self.maxRadius = maxRadius
        self.xBody = xBody
        self.yBody = yBody
        self.attraction = attraction
        self.divisionsPerSimulation = divisionsPerSimulation
        self.epochs = epochs
        self.angle = angle
        
        if self.maxSpeed <= 0:
            self.speedWeight = -self.maxSpeed
        
        self.loss = None
        self.lossValue = torch.tensor(0.0)
        self.lossHistory = []
        self.bias = 1
    
        self.model = TripleReluModel(nodes, minSpeed, maxSpeed, minTime, maxTime, angle)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learningRate)
        
        if load:
            load_status = self.load_state() 
            if not load_status.startswith("S"):
                print(f"Falha ao carregar o modelo: {load_status}. Iniciando um novo modelo.")
        
    def show_current_state(self, epoch, percent):
        statusString = f"Epoch: {epoch} ({percent:.1f}%)"
        print(f"{statusString:<25} |   Loss: {self.lossValue.item():.6f}")
        
    def simulate_launch(self, v, th, t, x, y):
        xi = torch.zeros_like(x)
        yi = torch.zeros_like(y)
        dt = t / self.divisionsPerSimulation
        vx = v * torch.cos(th)
        vy = v * torch.sin(th)
        
        for _ in range(self.divisionsPerSimulation):
            rVec_x = self.xBody - xi
            rVec_y = self.yBody - yi
            dist_sq = rVec_x**2 + rVec_y**2 + self.bias
            dist_cubed = dist_sq.pow(1.5)
            
            ax = (self.attraction * rVec_x) / dist_cubed
            ay = (self.attraction * rVec_y) / dist_cubed
            
            vx = vx + ax * dt
            vy = vy + ay * dt
            xi = xi + vx * dt
            yi = yi + vy * dt

        losses_tensor = (xi - x)**2 + (yi - y)**2 + (0 if self.maxSpeed > 0 else self.speedWeight * v**2)
        self.loss = torch.mean(losses_tensor)
        self.lossValue = self.loss.detach()
        
    def calculate_loss(self):
        N = self.pointsPerLoss
        alpha = torch.rand(N, 1) * (2 * math.pi)
        r = torch.sqrt(torch.rand(N, 1)) * self.maxRadius
        x_target = self.xBody + r * torch.cos(alpha)
        y_target = self.yBody + r * torch.sin(alpha)
        xy_targets = torch.cat([x_target, y_target], dim=1)
        
        v_pred, th_pred, t_pred = self.model(xy_targets)
        self.simulate_launch(v_pred, th_pred, t_pred, x_target, y_target)
        
    def fit(self, epochs = None, silent = False):
        if epochs is None:
            epochs = self.epochs
            
        self.model.train()
        print(f"Iniciando treino com {epochs} épocas.")
        t0 = time.time()
        
        for epoch in range(1, epochs + 1):
            self.calculate_loss()
            self.optimizer.zero_grad()
            
            if torch.isnan(self.loss):
                continue
            
            self.loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.lossHistory.append(self.lossValue.item())
            
            if (epoch % 500 == 0 or epoch == epochs) and not silent:
                self.show_current_state(epoch,(epoch/epochs)*100)
        
        tn = time.time()
        
        def convert_time(timeSpent):
            total_sec = np.floor(timeSpent)
            total_mins, z = divmod(total_sec, 60)
            x, y = divmod(total_mins, 60)
            
            t1 = "" if x == 0 else f'{x} hora' + ('s ' if x > 1 else ' ')
            t2 = "" if y == 0 else f'{y} minuto' + ('s ' if y > 1 else ' ')
            t3 = "" if z == 0 else f'{z} segundo' + ('s ' if z > 1 else ' ')
            
            return f"{t1}{t2}{t3}"
        
        print(f"Treino finalizado em {convert_time(tn-t0)}.")
        
    def predict(self, x_test):
        self.model.eval()
        x_tensor = torch.tensor(x_test, dtype=torch.float32)
        
        with torch.no_grad():
            v_pred, th_pred, t_final_pred = self.model(x_tensor)
        
        predictions = []
        for i, (xt, yt) in enumerate(x_test):
            predictions.append({
                "target": (xt, yt),
                "v": v_pred[i].item(),
                "theta": th_pred[i].item(),
                "t_final": t_final_pred[i].item()
            })
        return predictions
    
    def get_trajectory(self, v, th, t):
        v = torch.tensor(v, dtype=torch.float32)
        th = torch.tensor(th, dtype=torch.float32)
        t = torch.tensor(t, dtype=torch.float32)

        xi = torch.tensor(0.0)
        yi = torch.tensor(0.0)
        dt = t / self.divisionsPerSimulation
        vx = v * torch.cos(th)
        vy = v * torch.sin(th)
        
        trajectory = np.zeros((self.divisionsPerSimulation + 1, 2))
        trajectory[0] = [xi.item(), yi.item()]
        
        for i in range(self.divisionsPerSimulation):
            r_vec_x = self.xBody - xi
            r_vec_y = self.yBody - yi
            dist_sq = r_vec_x**2 + r_vec_y**2 + self.bias
            dist_cubed = dist_sq.pow(1.5)
            
            ax = (self.attraction * r_vec_x) / dist_cubed
            ay = (self.attraction * r_vec_y) / dist_cubed
            
            vx = vx + ax * dt
            vy = vy + ay * dt
            xi = xi + vx * dt
            yi = yi + vy * dt
            
            trajectory[i+1] = [xi.item(), yi.item()]
            
        return trajectory
    
    def plot_loss(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.lossHistory)
        plt.title('Histórico de Perda (Loss) do Treinamento')
        plt.xlabel('Época (Epoch)')
        plt.ylabel('Loss (Log Scale)')
        plt.yscale('log')
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.show()
        
    def plot_predictions(self, predictions, silent = True, legend = False):
        plt.figure(figsize=(12, 9))
        num_preds = len(predictions)
        colors = plt.cm.viridis(np.linspace(0, 1, num_preds))
    
        for i, pred in enumerate(predictions):
            target = pred["target"]
            v = pred["v"]
            th = pred["theta"]
            t = pred["t_final"]
            
            if not silent:
                print(f"  Alvo {target}: v={v:.2f}, theta={(th*180/math.pi):.2f}°, t={t:.2f}s")
            
            traj = self.get_trajectory(v, th, t)
            label = f'Alvo {target}'
            
            plt.plot(traj[:, 0], traj[:, 1], color=colors[i], linestyle='-', label=label)
            plt.scatter([target[0]], [target[1]], color=colors[i], s=100, zorder=5, marker='x')

        plt.scatter([0], [0], color='green', s=200, zorder=6, label='Origem')
        plt.scatter([self.xBody], [self.yBody], 
                    color='black', s=400, zorder=6, label='Planeta')
        
        plt.title(f'Previsões do Modelo')
        plt.xlabel('Posição X')
        plt.ylabel('Posição Y')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.axhline(0, color='grey', linewidth=0.5)
        plt.axvline(0, color='grey', linewidth=0.5)
        if legend:
            plt.legend(loc='best')
        plt.axis('equal')
        plt.show()
        
    def save_state(self):
        filepath = f'./models/gravitational/{self.identifier}.pth'
        
        state = state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_history': self.lossHistory,

            'identifier': self.identifier,
            'nodes': self.nodes,
            'minSpeed': self.minSpeed,
            'maxSpeed': self.maxSpeed,
            'minTime': self.minTime,
            'maxTime': self.maxTime,
            'learningRate': self.learningRate,
            'pointsPerLoss': self.pointsPerLoss,
            'maxRadius': self.maxRadius,
            'xBody': self.xBody,
            'yBody': self.yBody,
            'attraction': self.attraction,
            'divisionsPerSimulation': self.divisionsPerSimulation,
            'epochs': self.epochs,
            'angle': self.angle
        }
        
        torch.save(state, filepath)
        print(f"Modelo {self.identifier} salvo com sucesso em: '{filepath}'")

    def load_state(self):
        try:
            filepath = f'./models/gravitational/{self.identifier}.pth'
            savepoint = torch.load(filepath)
            savepoint['angle'] = self.angle
            
            self.model.load_state_dict(savepoint['model_state_dict'])
            self.optimizer.load_state_dict(savepoint['optimizer_state_dict'])
            self.lossHistory = savepoint['loss_history']
            
            self.nodes = savepoint['nodes']
            self.minSpeed = savepoint['minSpeed']
            self.maxSpeed = savepoint['maxSpeed']
            self.minTime = savepoint['minTime']
            self.maxTime = savepoint['maxTime']
            self.learningRate = savepoint['learningRate']
            self.pointsPerLoss = savepoint['pointsPerLoss']
            self.maxRadius = savepoint['maxRadius']
            self.xBody = savepoint['xBody']
            self.yBody = savepoint['yBody']
            self.attraction = savepoint['attraction']
            self.divisionsPerSimulation = savepoint['divisionsPerSimulation']
            self.epochs = savepoint['epochs']
            self.angle = savepoint['angle']
            
            self.model.eval()
            
            print(f"Modelo {self.identifier} carregado com sucesso de: '{filepath}'")
            if self.lossHistory:
                self.lossValue = torch.tensor(self.lossHistory[-1])
            return "S"
        except FileNotFoundError:
            return "N-File Not Found"
        except Exception as e:
            return f"E-{e}"
        
    def generate_random_points(self, N):
        alpha = np.random.rand(N, 1) * (2 * math.pi)
        r = np.sqrt(np.random.rand(N, 1)) * self.maxRadius
        x_target = self.xBody + r * np.cos(alpha)
        y_target = self.yBody + r * np.sin(alpha)
        return np.hstack([x_target, y_target])

    def evaluate(self, N):
        x_k = self.generate_random_points(N)
        preds = self.predict(x_k)
        self.plot_predictions(preds)

        all_errors = []
        for pred in preds:
            v = pred["v"]
            th = pred["theta"]
            t = pred["t_final"]
            target_pos = pred["target"]

            traj = self.get_trajectory(v, th, t)
            final_pos = traj[-1]
            
            error_dist = np.sqrt((final_pos[0] - target_pos[0])**2 + (final_pos[1] - target_pos[1])**2)
            all_errors.append(error_dist)
            
        errors_np = np.array(all_errors)
        mse = np.mean(np.square(errors_np))
        mean_error = np.mean(errors_np)
        min_error = np.min(errors_np)
        max_error = np.max(errors_np)
        median_error = np.median(errors_np)

        print(f"Erro Quadrático Médio (MSE): {mse:.4f}")
        print(f"Erro Médio:                 {mean_error:.4f}")
        print(f"Erro Mediano:               {median_error:.4f}")
        print(f"Erro Mínimo:                {min_error:.4f}")
        print(f"Erro Máximo:                {max_error:.4f}")
        
solver = TrajectoryPredictor(
    identifier = selected_model['identifier'],
    nodes = selected_model['nodes'],
    minSpeed = selected_model['minSpeed'],
    maxSpeed = selected_model['maxSpeed'],
    minTime = selected_model['minTime'],
    maxTime = selected_model['maxTime'],
    learningRate = selected_model['learningRate'],
    pointsPerLoss = selected_model['pointsPerLoss'],
    maxRadius = selected_model['maxRadius'],
    xBody = selected_model['xBody'],
    yBody = selected_model['yBody'],
    attraction = selected_model['attraction'],
    divisionsPerSimulation = selected_model['divisionsPerSimulation'],
    epochs = selected_model['epochs']
)

tryLoad = solver.load_state()
if tryLoad[0] != 'S':
    solver.fit()
    solver.save_state()
    
# --------------------------------------------------------------------

import pygame
import sys
import math
import imageio.v3 as iio

SECONDS = 1000

STATE_AIMING = "AIMING"
STATE_SIMULATING = "SIMULATING"

COLOR_BG = (10, 20, 30)
COLOR_CANNON = (200, 200, 200)
COLOR_BALL = (255, 100, 100)
COLOR_TARGET = (100, 255, 100)
COLOR_ARROW = (255, 255, 0)
COLOR_PREDICTION = (0, 255, 0, 100)
COLOR_TEXT = (255, 255, 255)
COLOR_NEBULA = (180, 180, 255, 128)
COLOR_NEBULA_CORE = (220, 220, 255, 192)

class Simulator:
    def __init__(self, solver: TrajectoryPredictor, w = 1200, h = 700, outscreen_limit = 100, hit_radius = 1, pixel_scale = 10, simulation_limit = 10, dragMin = 20, dragMax = 150, nebula_radius = 0.5):
        self.solver = solver
        self.width = w
        self.height = h
        
        self.pixel_scale = pixel_scale
        self.simulation_limit = simulation_limit * SECONDS
        self.outscreen_limit = outscreen_limit
        
        self.hit_radius = hit_radius
        
        self.cannonX = self.width//4
        self.cannonY = self.height - self.height//4
        
        self.dragMin = dragMin
        self.dragMax = dragMax
        
        self.nebula_radius_world = nebula_radius
        self.nebula_pixels = int(self.nebula_radius_world * self.pixel_scale)
        self.nebula_pixels_core = int(self.nebula_pixels * 0.8)
        
        self.targetX = 16.0
        self.targetY = 12.0
        self.score = 0
        
        self.prediction_v = 0.0
        self.prediction_theta = 0.0
        self.prediction_arrow_end_pos = (0, 0)
        
        self.gif_frames_bg = []
        self.current_gif_frame_bg = 0
        self.last_gif_update_bg = 0
        self.gif_frame_duration_bg = 100
        
        self.portal_frames_entry = []
        self.portal_frames_singularity = []
        self.portal_frames_exit = []
        self.current_portal_frame = 0 
        self.last_portal_update_portal = 0
        self.portal_frame_duration = 50
        
        self.entry_exit_size = (100, 100)
        self.entry_exit_rect_size = pygame.Rect(0, 0, *self.entry_exit_size)
        
        base_singularity_size = 20
        base_attraction = 100.0
        scale_factor = math.sqrt(max(0.1, self.solver.attraction) / base_attraction)
        dynamic_size = max(20, int(base_singularity_size * scale_factor))
        
        self.singularity_size = (dynamic_size, dynamic_size)
        self.singularity_rect_size = pygame.Rect(0, 0, *self.singularity_size)
        
    def world_to_screen(self, x, y):
        px = self.cannonX + x * self.pixel_scale
        py = self.cannonY - y * self.pixel_scale
        return int(px), int(py)
    
    def screen_to_world(self, px, py):
        x = (px - self.cannonX) / self.pixel_scale
        y = (self.cannonY - py) / self.pixel_scale
        return x, y
    
    def dist(self, pA, pB):
        return math.sqrt((pA[0] - pB[0])**2 + (pA[1] - pB[1])**2)
    
    def generate_new_target_and_prediction(self):
        min_r = self.solver.maxRadius / 3
        max_r = self.solver.maxRadius
        
        alpha = np.random.uniform(0, 2 * math.pi)
        r_sq = np.random.uniform(min_r**2, max_r**2)
        r = math.sqrt(r_sq)
        
        self.targetX = self.solver.xBody + r * math.cos(alpha)
        self.targetY = self.solver.yBody + r * math.sin(alpha)
        
        prediction_list = self.solver.predict(np.array([[self.targetX, self.targetY]]))
        prediction = prediction_list[0]
        
        self.prediction_v = prediction['v']
        self.prediction_theta = prediction['theta']
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    if self.game_state == STATE_AIMING:
                        self.is_dragging = True

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1 and self.is_dragging:
                    self.is_dragging = False
                    self.game_state = STATE_SIMULATING
                    self.simulation_start = pygame.time.get_ticks()
                    
                    self.ball_vel[0] = self.launch_speed * math.cos(self.launch_angle)
                    self.ball_vel[1] = self.launch_speed * math.sin(self.launch_angle)
                    self.ball_pos = [0.0, 0.0]
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and self.game_state == STATE_AIMING:
                    self.is_dragging = False
                    self.game_state = STATE_SIMULATING
                    self.simulation_start = pygame.time.get_ticks()
                    
                    self.ball_vel[0] = self.prediction_v * math.cos(self.prediction_theta)
                    self.ball_vel[1] = self.prediction_v * math.sin(self.prediction_theta)
                    self.ball_pos = [0.0, 0.0]
                    
                    
    def game_logic(self):
        dt = self.clock.get_time() / 1000.0
        self.cannon_screen_pos = self.world_to_screen(0,0)
        
        if self.is_dragging:
            mouse_pos = pygame.mouse.get_pos()
            dx_pix = mouse_pos[0] - self.cannon_screen_pos[0]
            dy_pix = mouse_pos[1] - self.cannon_screen_pos[1]
            
            drag_dist_pix = math.sqrt(dx_pix**2 + dy_pix**2)
            self.launch_angle = math.atan2(-dy_pix, dx_pix)
            
            drag_clamped = max(self.dragMin, min(self.dragMax, drag_dist_pix))
            
            speed_percent = (drag_clamped - self.dragMin) / (self.dragMax - self.dragMin)
            self.launch_speed = self.solver.minSpeed + speed_percent * (self.solver.maxSpeed - self.solver.minSpeed)
            
            arrow_end_x = self.cannon_screen_pos[0] + drag_clamped * math.cos(self.launch_angle)
            arrow_end_y = self.cannon_screen_pos[1] - drag_clamped * math.sin(self.launch_angle)
            self.arrow_end_pos = (int(arrow_end_x), int(arrow_end_y))
        
        elif self.game_state == STATE_SIMULATING:
            xi, yi = self.ball_pos[0], self.ball_pos[1]
            vx, vy = self.ball_vel[0], self.ball_vel[1]
            
            rVec_x = self.solver.xBody - xi
            rVec_y = self.solver.yBody - yi
            dist_sq = rVec_x**2 + rVec_y**2 + self.solver.bias
            dist_cubed = dist_sq * math.sqrt(dist_sq)
            
            ax = (self.solver.attraction * rVec_x) / dist_cubed
            ay = (self.solver.attraction * rVec_y) / dist_cubed
            
            vx += ax * dt
            vy += ay * dt
            xi += vx * dt
            yi += vy * dt
            
            self.ball_pos = [xi, yi]
            self.ball_vel = [vx, vy]
            
            if self.dist(self.ball_pos, (self.targetX, self.targetY)) < self.hit_radius:
                self.score += 1
                self.game_state = STATE_AIMING
                self.ball_pos = [0.0, 0.0]
                self.ball_vel = [0.0, 0.0]
                self.generate_new_target_and_prediction()
                
            elif (pygame.time.get_ticks() - self.simulation_start) > self.simulation_limit:
                self.score -= 1
                self.game_state = STATE_AIMING
                self.ball_pos = [0.0, 0.0]
                self.ball_vel = [0.0, 0.0]
            
            else:
                ball_screen_x, ball_screen_y = self.world_to_screen(self.ball_pos[0], self.ball_pos[1])
                if (ball_screen_x < -self.outscreen_limit or
                    ball_screen_x > self.width + self.outscreen_limit or
                    ball_screen_y < -self.outscreen_limit or
                    ball_screen_y > self.height + self.outscreen_limit):
                    
                    self.score -= 1
                    self.game_state = STATE_AIMING
                    self.ball_pos = [0.0, 0.0]
                    self.ball_vel = [0.0, 0.0]
                    
    def draw(self):
        self.prediction_surface.fill((0, 0, 0, 0))
        
        if self.gif_frames_bg:
            now = pygame.time.get_ticks()
            if now - self.last_gif_update_bg > self.gif_frame_duration_bg:
                self.current_gif_frame_bg = (self.current_gif_frame_bg + 1) % len(self.gif_frames_bg)
                self.last_gif_update_bg = now
            
            self.screen.blit(self.gif_frames_bg[self.current_gif_frame_bg], (0, 0))
        else:
            self.screen.fill(COLOR_BG)
        
        if self.portal_frames_entry and self.portal_frames_singularity and self.portal_frames_exit:
            now = pygame.time.get_ticks()
            if now - self.last_portal_update_portal > self.portal_frame_duration:
                self.current_portal_frame += 1 
                self.last_portal_update_portal = now
            
            frame_entry = self.portal_frames_entry[self.current_portal_frame % len(self.portal_frames_entry)]
            frame_singularity = self.portal_frames_singularity[self.current_portal_frame % len(self.portal_frames_singularity)]
            frame_exit = self.portal_frames_exit[self.current_portal_frame % len(self.portal_frames_exit)]

            if self.solver.xBody == self.solver.yBody:
                pos_grav = self.world_to_screen(self.solver.xBody, self.solver.yBody)
                rect_grav = self.singularity_rect_size.copy()
                rect_grav.center = pos_grav
                self.screen.blit(frame_singularity, rect_grav)
            
            pos_target = self.world_to_screen(self.targetX, self.targetY)
            rect_target = self.entry_exit_rect_size.copy()
            rect_target.center = pos_target
            self.screen.blit(frame_exit, rect_target)
            
            rect_start = self.entry_exit_rect_size.copy()
            rect_start.center = self.cannon_screen_pos
            self.screen.blit(frame_entry, rect_start)
            
        else:
            self.screen.blit(self.nebula, self.nebula_blit_pos)
            target_screen_pos = self.world_to_screen(self.targetX, self.targetY)
            hit_radius_pix = int(self.hit_radius * self.pixel_scale)
            pygame.draw.circle(self.screen, COLOR_TARGET, target_screen_pos, hit_radius_pix, 2)
            pygame.draw.circle(self.screen, COLOR_CANNON, self.cannon_screen_pos, 8)
        
        
        if self.game_state == STATE_AIMING:
            speed_percent = (self.prediction_v - self.solver.minSpeed) / (self.solver.maxSpeed - self.solver.minSpeed)
            drag_clamped = self.dragMin + speed_percent * (self.dragMax - self.dragMin)
            drag_dist = max(self.dragMin, min(self.dragMax, drag_clamped))
            
            pred_arrow_end_x = self.cannon_screen_pos[0] + drag_dist * math.cos(self.prediction_theta)
            pred_arrow_end_y = self.cannon_screen_pos[1] - drag_dist * math.sin(self.prediction_theta)
            self.prediction_arrow_end_pos = (int(pred_arrow_end_x), int(pred_arrow_end_y))
            
            pygame.draw.line(self.prediction_surface, COLOR_PREDICTION, self.cannon_screen_pos, self.prediction_arrow_end_pos, 3)

            if self.is_dragging:
                pygame.draw.line(self.screen, COLOR_ARROW, self.cannon_screen_pos, self.arrow_end_pos, 3)
                speed_text = self.font_small.render(f"Speed: {self.launch_speed:.1f}", True, COLOR_TEXT)
                self.screen.blit(speed_text, (10, 10))

        elif self.game_state == STATE_SIMULATING:
            ball_screen_pos = self.world_to_screen(self.ball_pos[0], self.ball_pos[1])
            pygame.draw.circle(self.screen, COLOR_BALL, ball_screen_pos, 5)

        score_text_surf = self.font_small.render(f"Score: {self.score}", True, COLOR_TEXT)
        score_text_rect = score_text_surf.get_rect(topright=(self.width - 10, 10))
        self.screen.blit(score_text_surf, score_text_rect)
        self.screen.blit(self.prediction_surface, (0, 0))
    
    def initialize(self):
        pygame.init()
        pygame.font.init()

        self.screen = pygame.display.set_mode((self.width, self.height))
        self.prediction_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        
        pygame.display.set_caption("Angry... Asteroids?")
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Arial", 48)
        self.font_small = pygame.font.SysFont("Arial", 24)
        
        if iio:
            try:
                gif_path = 'res/background.gif'
                gif_frames_raw = iio.imiter(gif_path, plugin="pillow")
                
                for frame in gif_frames_raw:
                    mode = "RGB" if frame.shape[2] == 3 else "RGBA"
                    frame_surface = pygame.image.frombytes(frame.tobytes(), frame.shape[1::-1], mode)
                    frame_surface = pygame.transform.scale(frame_surface, (self.width, self.height))
                    self.gif_frames_bg.append(frame_surface)
            except Exception as e:
                print(f"Erro ao carregar background.gif: {e}")
                self.gif_frames_bg = [] 

        if iio:
            try:
                gif_path_entry = 'res/portal.gif'
                gif_frames_raw_entry = iio.imiter(gif_path_entry, plugin="pillow")
                self.portal_frames_entry = []
                for frame in gif_frames_raw_entry:
                    mode = "RGB" if frame.shape[2] == 3 else "RGBA"
                    frame_surface = pygame.image.frombytes(frame.tobytes(), frame.shape[1::-1], mode).convert_alpha()
                    frame_surface = pygame.transform.scale(frame_surface, self.entry_exit_size)
                    self.portal_frames_entry.append(frame_surface)
                
                if not self.portal_frames_entry:
                    print(f"Warning: Portal GIF '{gif_path_entry}' loaded 0 frames or failed.")

                gif_path_singularity = 'res/singularity.gif'
                gif_frames_raw_singularity = iio.imiter(gif_path_singularity, plugin="pillow")
                self.portal_frames_singularity = []
                for frame in gif_frames_raw_singularity:
                    mode = "RGB" if frame.shape[2] == 3 else "RGBA"
                    frame_surface = pygame.image.frombytes(frame.tobytes(), frame.shape[1::-1], mode).convert_alpha()
                    frame_surface = pygame.transform.scale(frame_surface, self.singularity_size)
                    self.portal_frames_singularity.append(frame_surface)
                    
                gif_path_exit = 'res/red_portal.gif'
                gif_frames_raw_exit = iio.imiter(gif_path_exit, plugin="pillow")
                self.portal_frames_exit = []
                for frame in gif_frames_raw_exit:
                    mode = "RGB" if frame.shape[2] == 3 else "RGBA"
                    frame_surface = pygame.image.frombytes(frame.tobytes(), frame.shape[1::-1], mode).convert_alpha()
                    frame_surface = pygame.transform.scale(frame_surface, self.entry_exit_size)
                    self.portal_frames_exit.append(frame_surface)
            
            except Exception as e:
                 print(f"Erro ao carregar GIFs do portal: {e}")
                 self.portal_frames_entry = []
                 self.portal_frames_singularity = []
                 self.portal_frames_exit = []


        self.nebula = pygame.Surface((self.nebula_pixels * 2, self.nebula_pixels * 2), pygame.SRCALPHA)
        pygame.draw.circle(self.nebula, COLOR_NEBULA, (self.nebula_pixels, self.nebula_pixels), self.nebula_pixels)
        pygame.draw.circle(self.nebula, COLOR_NEBULA_CORE, (self.nebula_pixels, self.nebula_pixels), self.nebula_pixels_core)
        
        self.nebula_pos = self.world_to_screen(self.solver.xBody, self.solver.yBody)
        self.nebula_blit_pos = (self.nebula_pos[0] - self.nebula_pixels, self.nebula_pos[1] - self.nebula_pixels)
        
        self.game_state = STATE_AIMING
        self.simulation_start = 0
        self.is_dragging = False

        self.ball_pos = [0.0, 0.0]
        self.ball_vel = [0.0, 0.0]

        self.launch_speed = 0.0
        self.launch_angle = 0.0
        self.arrow_end_pos = (0, 0)
        
        self.generate_new_target_and_prediction()
        
        self.running = True
        while self.running:
            self.handle_events()
            self.game_logic()
            self.draw()
            pygame.display.flip()
            self.clock.tick(60)

sim = Simulator(solver)
sim.initialize()
pygame.quit()
sys.exit()