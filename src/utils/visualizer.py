# Visualização de resultados e dados obtidos em treinamentos
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import List
from src.core.predictor import TrajectoryPredictor

class Visualizer:
    """
    Classe responsável pela visualização de resultados, métricas de erro 
    e renderização de gráficos de trajetórias obtidas pelo modelo.

    Args:
        predictor (TrajectoryPredictor): Instância do preditor contendo o modelo e pesos carregados.
    """

    def __init__(self, predictor: TrajectoryPredictor):
        self.p = predictor

    def plot_loss(self) -> None:
        """
        Plota o histórico de perda do treinamento em escala logarítmica utilizando Matplotlib.

        Returns:
            None: Exibe o gráfico na interface padrão.
        """
        if self.p.hp['stage'] == 0:
            print('Essa ação só pode ser executada após o treino')
            return
        
        self.p.architecture.eval()
        plt.figure(figsize=(10, 5))
        plt.plot(self.p.lossHistory)
        plt.title('Histórico de Perda do Treinamento')
        plt.xlabel('Época')
        plt.ylabel('Perda (Escala Log)')
        plt.yscale('log')
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.show()

    def predict(self, N: int|List[List[float]]) -> None:
        """
        Gera predições para alvos aleatórios ou específicos, calcula métricas de erro 
        e plota as trajetórias resultantes comparadas aos alvos.

        Args:
            N (int|List[List[float]]): Se int, gera N alvos aleatórios. 
                Se lista, utiliza as coordenadas fornecidas no formato [[x, y], ...].

        Returns:
            None: Exibe métricas no console e o gráfico de trajetórias.
        """
        
        if self.p.hp['stage'] == 0:
            print('Essa ação só pode ser executada após o treino')
            return
        
        self.p.architecture.eval()
        
        x_target: torch.Tensor
        y_target: torch.Tensor
        input_values: torch.Tensor

        if isinstance(N, int):
            num_samples = N
            alpha = torch.rand((num_samples, 1), device=self.p.hp['device']) * (2 * math.pi)
            r = torch.sqrt(torch.rand((num_samples, 1), device=self.p.hp['device'])) * self.p.hp['maxDistance']
            
            hab_indxs = torch.tensor(
                [i for i, b in enumerate(self.p.hp['bodies']) if b.get('habitable', True)], 
                device=self.p.hp['device']
            )
            indxs_raw = torch.randint(0, len(hab_indxs), (num_samples,), device=self.p.hp['device'])
            indxs = hab_indxs[indxs_raw]
            
            x_target = r * torch.cos(alpha) + self.p.bX[indxs].view(num_samples, 1)
            y_target = r * torch.sin(alpha) + self.p.bY[indxs].view(num_samples, 1)
            input_values = torch.cat([x_target, y_target], dim=1)
            
        elif isinstance(N, list):
            input_values = torch.tensor(N, device=self.p.hp['device'], dtype=torch.float32)
            x_target = input_values[:, 0:1]
            y_target = input_values[:, 1:2]
            num_samples = len(N)
        else:
            raise ValueError("N deve ser um inteiro ou uma lista de coordenadas [[x, y], ...]")

        v_, th_, t_ = self.p.architecture(input_values)
        xTrajs, yTrajs, losses = self.p.lossFuncTraj(
            v_, th_, t_, x_target, y_target, self.p.bX, self.p.bY, self.p.bMass, 
            self.p.hp['bias'], self.p.hp['stepsPerSimulation']
        )

        print(f"MSE:                  {torch.mean(losses):.6f}")
        print(f"MAE:                  {torch.mean(torch.abs(losses**0.5)):.6f}")
        print(f"Erro Mínimo:          {torch.min(losses**0.5):.6f}")
        print(f"Erro Máximo:          {torch.max(losses**0.5):.6f}")
        print(f"Mediana dos Erros:    {torch.median(losses**0.5):.6f}")

        plt.figure(figsize=(12, 9))
        colors = plt.colormaps['viridis'](np.linspace(0, 1, num_samples))

        x_np = x_target.detach().cpu().numpy()
        y_np = y_target.detach().cpu().numpy()

        for i in range(num_samples):
            plt.plot(xTrajs[:, i], yTrajs[:, i], color=colors[i], alpha=0.6, linewidth=1, 
                     label=f'Obj {i}' if num_samples <= 10 else None)
            plt.scatter(x_np[i][0], y_np[i][0], color=colors[i], s=20)

        bx = self.p.bX.detach().cpu().numpy()
        by = self.p.bY.detach().cpu().numpy()
        plt.scatter(bx, by, c='red', marker='x', s=100, label='Corpos Gravitacionais')
        plt.title(f'Predição de Trajetórias (N={num_samples})')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True, linestyle=':')
        plt.show()