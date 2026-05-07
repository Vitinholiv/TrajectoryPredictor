# Guarda as arquiteturas gerais utilizadas
import torch
import torch.nn as nn

class SingleAim(nn.Module):
    """
    Rede Neural Multi-Layer Perceptron projetada para prever os parâmetros de 
    lançamento de um projétil em um ambiente com gravidade simulada.

    A arquitetura recebe as coordenadas de um alvo bidimensional e prevê a velocidade 
    inicial (v), o ângulo de lançamento (theta) e o tempo de voo (t). As saídas são 
    automaticamente escalonadas para intervalos físicos limitados utilizando a função 
    Sigmoid.

    Args:
        layers (list): Lista de inteiros definindo o número de neurônios de cada camada oculta.
        activations (list): Lista de funções de ativação (como nn.ReLU()) respectivamente às camadas.
        limSpeed (list): Intervalo permitido para a velocidade de saída no formato [min, max].
        limTime (list): Intervalo permitido para o tempo de voo no formato [min, max].
        limAngle (list): Intervalo permitido para o ângulo no formato [min, max] (em radianos).
    """

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
        
    def forward(self, inputTensor:torch.Tensor):
        """
        Forward pass da arquitetura completa.

        Args:
            inputTensor (torch.Tensor): Tensor de formato (N, 2), onde N é o tamanho do batch e 2 representa as coordenadas (x, y) do alvo.

        Returns:
            Output: Três tensores de formato (N, 1) representando v, theta e t, respectivamente.
        """
        
        outputs = self.network(inputTensor)
        v = self.limSpeed[0] + torch.sigmoid(outputs[:, 0:1])*(self.limSpeed[1] - self.limSpeed[0])
        theta = self.limAngle[0] + torch.sigmoid(outputs[:, 1:2])*(self.limAngle[1] - self.limAngle[0])
        t = self.limTime[0] + torch.sigmoid(outputs[:, 2:3])*(self.limTime[1] - self.limTime[0])
        return v, theta, t