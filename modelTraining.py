import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='pygame.pkgdata')

import os
import json
import time
import copy
import math
from datetime import datetime

import matplotlib.pyplot as plt
import scipy.constants as const
import numpy as np
import torch
import torch.nn as nn

import pygame
import imageio.v3 as iio

# ---------- Arquiteturas ----------

class SingleAim(nn.Module):
    def __init__(self, layers:list, activations:list, limSpeed:list, limTime:list, limAngle:list):
        super().__init__()
        self.name = 'SingleAim'
        self.limSpeed = limSpeed
        self.limTime = limTime
        self.limAngle = limAngle
        
        arch = []
        arch.append(nn.Linear(2,layers[0]))
        arch.append(activations[0])
        for i in range(1,len(layers)):
            arch.append(nn.Linear(layers[i-1],layers[i]))
            arch.append(activations[i])
        arch.append(nn.Linear(layers[len(layers)-1],3))
        self.network = nn.Sequential(*arch)
        
    def forward(self, inputTensor):
        outputs = self.network(inputTensor)
        v = self.limSpeed[0] + torch.sigmoid(outputs[:, 0:1])*(self.limSpeed[1] - self.limSpeed[0])
        theta = self.limAngle[0] + torch.sigmoid(outputs[:, 1:2])*(self.limAngle[1] - self.limAngle[0])
        t = self.limTime[0] + torch.sigmoid(outputs[:, 2:3])*(self.limTime[1] - self.limTime[0])
        return v, theta, t

# ---------- Perdas ----------

def loss_euler_step(bX, bY, bMass, xi, yi, vx, vy, bias, dt):
    rVec_x = bX.view(1, -1) - xi
    rVec_y = bY.view(1, -1) - yi
    dist_sq = rVec_x**2 + rVec_y**2
    dist_cubed = (dist_sq + bias**2)**1.5 # Plummer Softening

    ax = torch.sum((const.G * bMass.view(1, -1) * rVec_x)/dist_cubed, dim=1, keepdim=True)
    ay = torch.sum((const.G * bMass.view(1, -1) * rVec_y)/dist_cubed, dim=1, keepdim=True)
    
    vx = vx + ax * dt
    vy = vy + ay * dt
    xi = xi + vx * dt
    yi = yi + vy * dt
    return xi,yi,vx,vy

def loss_euler(v,th,t, x,y, bX,bY,bMass, bias,steps):
    xi = torch.zeros_like(x)
    yi = torch.zeros_like(y)
    dt = t / steps
    vx = v * torch.cos(th)
    vy = v * torch.sin(th)

    for _ in range(steps):
        xi,yi,vx,vy = loss_euler_step(bX,bY,bMass,xi,yi,vx,vy,bias,dt)

    loss = (xi - x)**2 + (yi - y)**2
    return torch.mean(loss)

def loss_euler_trajectory(v,th,t, x,y, bX,bY,bMass, bias,steps):
    xi = torch.zeros_like(x)
    yi = torch.zeros_like(y)
    dt = t / steps
    vx = v * torch.cos(th)
    vy = v * torch.sin(th)
    
    xCoords = [xi.clone()]
    yCoords = [yi.clone()]

    for _ in range(steps):
        xi,yi,vx,vy = loss_euler_step(bX,bY,bMass,xi,yi,vx,vy,bias,dt)
        xCoords.append(xi.clone())
        yCoords.append(yi.clone())

    loss = ((xi - x)**2 + (yi - y)**2)**0.5
    return torch.stack(xCoords).detach().numpy(), torch.stack(yCoords).detach().numpy(), loss.detach().cpu()

# ---------- Modelo ----------

class TrajectoryPredictor:
    def __init__(self, hp:dict):
        self.hp = hp.copy()

        if self.hp['lossFunc'] == 'loss_euler':
            self.lossFunc = loss_euler
            self.lossFuncTraj = loss_euler_trajectory
        else:
            raise ValueError("Função de perda inválida")

        if self.hp['architecture'] == 'SingleAim':
            self.architecture = SingleAim(self.hp['layers'], self.hp['activations'], self.hp['limSpeed'], self.hp['limTime'], self.hp['limAngle'])
            self.architecture.to(self.hp['device'])
            self.bX = torch.tensor([b['x'] for b in self.hp['bodies']], device=self.hp['device'])
            self.bY = torch.tensor([b['y'] for b in self.hp['bodies']], device=self.hp['device'])
            self.bMass = torch.tensor([b['mass'] for b in self.hp['bodies']], device=self.hp['device'])
        else:
            raise ValueError("Arquitetura inválida")

        if self.hp['optimizer'] == 'Adam':
            self.optimizer = torch.optim.Adam(self.architecture.parameters(), lr=self.hp['learningRate'])
        else:
            raise ValueError("Otimizador inválido")
        
        self.bestLossValue = float('inf')
        self.bestLoss = torch.tensor([])
        self.bestState = {}
        self.loss = torch.tensor(0.0)
        self.lossHistory = []

    def save(self):
        filepath = f"./models/{self.hp['id']}/"
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filenamePth = timestamp + '.pth'
        filenameJson = timestamp + '.json'
        thisfilePth = os.path.join(filepath, filenamePth)
        thisfileJson = os.path.join(filepath, filenameJson)

        if not os.path.exists(thisfilePth):
            state = {}
            state['hp'] = self.hp
            state['lossHistory'] = self.lossHistory

            state['architectureStateDict'] = self.bestState['architecture']
            state['optimizerStateDict'] = self.bestState['optimizer']
            state['loss'] = self.bestLoss

            saveHp = self.hp.copy()

            def sanitizeToJson(obj):
                if isinstance(obj, (np.integer, np.floating)):
                    return float(obj) if isinstance(obj, np.floating) else int(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, torch.Tensor):
                    return obj.detach().cpu().tolist()
                if isinstance(obj, list):
                    return [sanitizeToJson(x) for x in obj]
                if isinstance(obj, dict):
                    return {k: sanitizeToJson(v) for k, v in obj.items()}
                if isinstance(obj, (str, int, float, bool, type(None))):
                    return obj
                return str(obj)
            
            saveHp = sanitizeToJson(saveHp)
            with open(thisfileJson, 'w', encoding='utf-8') as f:
                json.dump(saveHp, f, indent=4, ensure_ascii=False)

            torch.save(state, thisfilePth)
            print(f"Modelo {self.hp['id']} salvo em: '{thisfilePth}' com especificações salvas em '{thisfileJson}'")
        else:
            print(f"Essa instância do modelo já existe")

    def load(self):
        filepath = f"./models/{self.hp['id']}/"
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        files = [f for f in os.listdir(filepath) if f.endswith('.pth')]

        if len(files) == 0:
            print('Nenhuma instância desse modelo existe, inicializando modelo vazio')
        else:
            files = sorted(files)
            print('Escolha qual versão do modelo carregar:')
            for i,fl in enumerate(files):
                print(f" ({i}) {fl}")
            print(f" (-) Modelo vazio")
            print(f" ( ) Modelo mais recente")

            res = input(); id = 0
            try:
                id = int(res)
                if id < 0:
                    print("Inicializando modelo vazio")
                    return
                else:
                    id %= len(files)
            except ValueError:
                if len(res) > 0:
                    print("Inicializando modelo vazio")
                    return
                else:
                    id = len(files)-1

            thisfile = os.path.join(filepath,files[id])
            try:
                savepoint = torch.load(thisfile,weights_only=False)
                self.hp = savepoint['hp']
                self.architecture.load_state_dict(savepoint['architectureStateDict'])
                self.optimizer.load_state_dict(savepoint['optimizerStateDict'])
                self.lossHistory = savepoint['lossHistory']
                self.loss = savepoint['loss']

                if self.architecture.name == "SingleAim":
                    self.bX = torch.tensor([b['x'] for b in self.hp['bodies']], device=self.hp['device'])
                    self.bY = torch.tensor([b['y'] for b in self.hp['bodies']], device=self.hp['device'])
                    self.bMass = torch.tensor([b['mass'] for b in self.hp['bodies']], device=self.hp['device'])
                else:
                    print('NOT IMPLEMENTED')
                print(f"Modelo {self.hp['id']} carregado de: '{thisfile}'")
            except Exception as e:
                print('Erro durante o carregamento: ',e)

    def calculate_loss(self):
        if self.architecture.name == 'SingleAim':
            N = self.hp['pointsPerLoss']
            alpha = torch.rand((N,1), device=self.hp['device']) * (2 * math.pi)
            r = torch.sqrt(torch.rand((N,1), device=self.hp['device'])) * self.hp['maxDistance']
            indxs = torch.randint(0, len(self.hp['bodies']), (N,), device=self.hp['device'])

            xTarget = r * torch.cos(alpha) + self.bX[indxs].view(N, 1)
            yTarget = r * torch.sin(alpha) + self.bY[indxs].view(N, 1)
            inputValues = torch.cat([xTarget, yTarget], dim=1)
            
            vPred, thPred, tPred = self.architecture(inputValues)
            self.loss = self.lossFunc(vPred, thPred, tPred, xTarget, yTarget, self.bX, self.bY, self.bMass, self.hp['bias'], self.hp['stepsPerSimulation'])
        else:
            print('NOT IMPLEMENTED')
            
    def fit(self, silent = False):
        self.architecture.train()
        print(f"Iniciando treino com {self.hp['epochs']} épocas.")
        t0 = time.time()

        for epoch in range(1, self.hp['epochs'] + 1):
            self.optimizer.zero_grad()
            self.calculate_loss()
            
            if torch.isnan(self.loss):
                continue
                
            if self.loss.item() < self.bestLossValue:
                self.bestLoss = self.loss.detach().clone()
                self.bestLossValue = self.loss.item()
                self.bestState = {
                    'architecture': copy.deepcopy(self.architecture.state_dict()),
                    'optimizer': copy.deepcopy(self.optimizer.state_dict())
                }
            
            self.loss.backward()
            torch.nn.utils.clip_grad_norm_(self.architecture.parameters(), max_norm=1.0)
            self.optimizer.step()

            self.lossHistory.append(self.loss.detach().cpu().item()) 
            
            if ( (epoch % 100 == 0) or (epoch == self.hp['epochs']) ) and not silent:
                statusString = f"Epoch: {epoch} ({((epoch/self.hp['epochs'])*100):.1f}%)"
                print(f"{statusString:<25} |   Loss = [Last Value: {self.loss.detach():.4f}, Last Mean: {np.mean(self.lossHistory[-100:]):.4f}, Last Max: {np.max(self.lossHistory[-100:]):.4f}, Last Min: {np.min(self.lossHistory[-100:]):.4f}]")
        
        tn = time.time()
        
        def convert_time(timeSpent):
            totalSec = np.floor(timeSpent)
            totalMins, z = divmod(totalSec, 60)
            x, y = divmod(totalMins, 60)
            
            t1 = "" if x == 0 else f'{x} hora' + ('s ' if x > 1 else ' ')
            t2 = "" if y == 0 else f'{y} minuto' + ('s ' if y > 1 else ' ')
            t3 = "" if z == 0 else f'{z} segundo' + ('s ' if z > 1 else ' ')
            
            return f"{t1}{t2}{t3}"
        
        self.hp['stage'] = 1
        print(f"Treino finalizado em {convert_time(tn-t0)}.")

# ---------- Versões ----------

defaultModel = {
    # Hiperparâmetros fixos - Devem ser fixos dado o modelo
    'id': 'defaultModel',
    'lossFunc': 'loss_euler',
    'architecture': 'SingleAim',
    'optimizer': 'Adam',
    'layers': [96, 96, 96],
    'activations': [nn.ReLU(), nn.ReLU(), nn.ReLU()],
    'limSpeed': [0, 20],
    'limTime': [0, 10],
    'limAngle': [0, 2*np.pi],
    'device': 'cpu',
    'stage': 0,
    # Hiperparâmetros variáveis - Um mesmo modelo pode ser executado com diferentes valores destes
    'epochs': 20000,
    'learningRate': 1e-4,
    'stepsPerSimulation': 200,
    'pointsPerLoss': 1024,
    'maxDistance': 20,
    'bodies': [
        {'x': 20, 'y': 20, 'mass': 1.1e14}
    ],
    'bias': 1
}

def create_model(id,changes):
    model = copy.deepcopy(defaultModel)
    model.update(changes)
    model['id'] = id
    return model

# ---------- Instâncias ----------

instance = TrajectoryPredictor(defaultModel)

instance.load()
if instance.hp['stage'] == 0:
    instance.fit()

if instance.hp['stage'] == 1:
    instance.hp['stage'] = 2
    instance.save()