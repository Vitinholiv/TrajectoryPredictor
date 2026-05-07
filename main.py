import os
import warnings

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
warnings.filterwarnings("ignore", category=UserWarning, module='pygame.pkgdata')
warnings.filterwarnings("ignore", category=DeprecationWarning)

from configs.default import defaultModel, dummyModel
from src.core.predictor import TrajectoryPredictor
from src.core.trainer import Trainer
from src.utils.visualizer import Visualizer
from src.utils.simulator import Simulator

if __name__ == "__main__":
    print('Inicializando...')
    predictor = TrajectoryPredictor(hp=dummyModel)
    predictor.load()

    trainer = Trainer(predictor=predictor)
    trainer.fit()
    predictor.save()
    
    simulator = Simulator(predictor)
    simulator.simulate()