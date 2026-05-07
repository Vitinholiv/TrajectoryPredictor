# Guarda a classe de instância de um preditor de trajetórias
import os
import json
from datetime import datetime
import numpy as np
import torch

from src.models.architectures import SingleAim
from src.core.losses import (
    loss_euler, loss_euler_trajectory,
    loss_runge_kutta, loss_runge_kutta_trajectory
)
from src.utils.physics import loss_euler_step, loss_runge_kutta_step

class TrajectoryPredictor:
    """
    Gerenciador de instância do modelo de predição de trajetórias.
    
    Classe responsável por manter o estado do modelo, incluindo sua arquitetura, 
    hiperparâmetros, pesos e histórico de treinamento. Atua como o núcleo de 
    persistência, lidando com a inicialização, salvamento e carregamento de instâncias.

    Args:
        hp (dict): Dicionário contendo os hiperparâmetros do modelo.
    """
    
    def __init__(self, hp: dict):
        self.hp = hp.copy()

        if self.hp['lossFunc'] == 'loss_euler':
            self.lossFunc = loss_euler
            self.lossFuncTraj = loss_euler_trajectory
            self.lossFuncStep = loss_euler_step
        elif self.hp['lossFunc'] == 'loss_runge_kutta':
            self.lossFunc = loss_runge_kutta
            self.lossFuncTraj = loss_runge_kutta_trajectory
            self.lossFuncStep = loss_runge_kutta_step
        else:
            raise ValueError("Função de perda inválida")

        if self.hp['architecture'] == 'SingleAim':
            self.architecture = SingleAim(
                self.hp['layers'], self.hp['activations'], 
                self.hp['limSpeed'], self.hp['limTime'], self.hp['limAngle']
            )
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
        """
        Salva o estado atual do modelo em arquivos de disco (.pth para pesos e .json para metadados).
        
        O arquivo é salvo em uma estrutura de pastas baseada no ID do modelo, utilizando 
        um timestamp para diferenciar as versões.

        Returns:
            None: Imprime o status do salvamento no console.
        """

        filepath = f"./runs/{self.hp['id']}/"
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
        """
        Carrega um estado salvo do modelo a partir de arquivos.
        
        Apresenta ao usuário uma lista de versões disponíveis para o ID do modelo atual, 
        permitindo a seleção de um checkpoint específico ou do mais recente.

        Returns:
            None: Atualiza o estado interno da classe com os dados carregados.
        """
        
        filepath = f"./runs/{self.hp['id']}/"
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
                    print("Inicializando modelo vazio"); return
                else:
                    id %= len(files)
            except ValueError:
                if len(res) > 0:
                    print("Inicializando modelo vazio"); return
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
                print(f"Modelo {self.hp['id']} carregado de: '{thisfile}'")
            except Exception as e:
                print('Erro durante o carregamento: ',e)