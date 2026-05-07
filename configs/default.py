import torch.nn as nn
import numpy as np
import copy

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
        {'x': 20, 'y': 20, 'mass': 1.1e14, 'habitable': True}
    ],
    'bias': 1
}

def create_model(id,changes):
    model = copy.deepcopy(defaultModel)
    model.update(changes)
    model['id'] = id
    return model

dummyModel = create_model("dummyModel", {
    'id': 'dummyModel',
    'lossFunc': 'loss_runge_kutta',
    'pointsPerLoss': 120,
    'bodies': [
        {'x': 10, 'y': 20, 'mass': 1.1e14, 'habitable': True},
        {'x': -17, 'y': 25, 'mass': 2.2e13, 'habitable': False}
    ],
    'epochs': 67
})