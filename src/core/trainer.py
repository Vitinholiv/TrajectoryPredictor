# Responsável pelo treino de modelos

import time
import copy
import math
import numpy as np
import torch

from src.core.predictor import TrajectoryPredictor
from src.utils.helpers import convert_time

class Trainer:
    """
    Classe responsável por gerenciar o processo de otimização e treinamento do modelo.
    
    O Trainer atua sobre uma instância de TrajectoryPredictor, executando ciclos de 
    treinamento, gerando dados sintéticos de alvos, calculando a perda 
    através da física e atualizando os pesos da rede neural.

    Args:
        predictor (TrajectoryPredictor): A instância do preditor que contém a 
            arquitetura e os hiperparâmetros a serem otimizados.
    """

    def __init__(self, predictor: TrajectoryPredictor):
        self.p = predictor

    def calculate_loss(self):
        """
        Gera amostras aleatórias de alvos ao redor de corpos habitáveis e calcula a 
        perda baseada na simulação física da trajetória.

        O método sorteia N pontos em coordenadas polares ao redor dos planetas, 
        realiza o forward pass na rede e invoca a função de perda configurada (Euler ou RK4).

        Returns:
            None: O resultado é armazenado diretamente no atributo self.p.loss.
        """

        if self.p.architecture.name == 'SingleAim':
            N = self.p.hp['pointsPerLoss']
            alpha = torch.rand((N,1), device=self.p.hp['device']) * (2 * math.pi)
            r = torch.sqrt(torch.rand((N,1), device=self.p.hp['device'])) * self.p.hp['maxDistance']

            hab_indxs = torch.tensor([ i for i, body in enumerate(self.p.hp['bodies']) if body.get('habitable', True) is True ],device=self.p.hp['device'])
            indxs = torch.randint(0, len(hab_indxs), (N,), device=self.p.hp['device'])
            true_indxs = hab_indxs[indxs]

            xTarget = r * torch.cos(alpha) + self.p.bX[true_indxs].view(N, 1)
            yTarget = r * torch.sin(alpha) + self.p.bY[true_indxs].view(N, 1)
            inputValues = torch.cat([xTarget, yTarget], dim=1)
            
            vPred, thPred, tPred = self.p.architecture(inputValues)
            self.p.loss = self.p.lossFunc(
                vPred, thPred, tPred, xTarget, yTarget, 
                self.p.bX, self.p.bY, self.p.bMass, 
                self.p.hp['bias'], self.p.hp['stepsPerSimulation']
            )

    def fit(self, silent:bool=False):
        """
        Executa o loop principal de treinamento do modelo.
        
        Realiza o ciclo de backpropagation, atualiza os pesos do otimizador, 
        aplica clipagem de gradiente e armazena o melhor estado (best state) 
        encontrado durante as épocas.

        Args:
            silent (bool): Se True, suprime os logs de progresso no console.

        Returns:
            None: Atualiza o histórico de loss e os pesos da arquitetura no predictor.
        """

        self.p.architecture.train()
        print(f"Iniciando treino com {self.p.hp['epochs']} épocas.")
        t0 = time.time()

        for epoch in range(1, self.p.hp['epochs'] + 1):
            self.p.optimizer.zero_grad()
            self.calculate_loss()
            
            if torch.isnan(self.p.loss) or torch.isinf(self.p.loss):
                continue
                
            if self.p.loss.item() < self.p.bestLossValue:
                self.p.bestLoss = self.p.loss.detach().clone()
                self.p.bestLossValue = self.p.loss.item()
                self.p.bestState = {
                    'architecture': copy.deepcopy(self.p.architecture.state_dict()),
                    'optimizer': copy.deepcopy(self.p.optimizer.state_dict())
                }
            
            self.p.loss.backward()
            torch.nn.utils.clip_grad_norm_(self.p.architecture.parameters(), max_norm=1.0)
            self.p.optimizer.step()

            self.p.lossHistory.append(self.p.loss.detach().cpu().item()) 
            
            if ((epoch % 100 == 0) or (epoch == self.p.hp['epochs'])) and not silent:
                statusString = f"Epoch: {epoch} ({((epoch/self.p.hp['epochs'])*100):.1f}%)"
                print(f"{statusString:<25} |   Loss = [Last Value: {self.p.loss.detach():.4f}, Last Mean: {np.mean(self.p.lossHistory[-100:]):.4f}, Last Max: {np.max(self.p.lossHistory[-100:]):.4f}, Last Min: {np.min(self.p.lossHistory[-100:]):.4f}]")
        
        if self.p.bestState:
            self.p.architecture.load_state_dict(self.p.bestState['architecture'])
            self.p.optimizer.load_state_dict(self.p.bestState['optimizer'])

        tn = time.time()
        self.p.hp['stage'] = 1
        print(f"Treino finalizado em {convert_time(tn-t0)}.")