# Arquivo feito para guardar funções auxiliares
import numpy as np

def convert_time(timeSpent: int|float) -> str:
    """
        Converte `timeSpent` segundos em uma string formatada de tempo com horas, minutos e segundos.

        Args:
            timeSpent (int|float): Valor em segundos do tempo para ser convertido.
        
        Returns:
            string: A string formatada.
    """

    timeSpent = int(timeSpent)
    totalSec = int(np.floor(timeSpent))
    totalMins, z = divmod(totalSec, 60)
    x, y = divmod(totalMins, 60)
    
    t1 = "" if x == 0 else f'{x} hora' + ('s ' if x > 1 else ' ')
    t2 = "" if y == 0 else f'{y} minuto' + ('s ' if y > 1 else ' ')
    t3 = "" if z == 0 else f'{z} segundo' + ('s ' if z > 1 else ' ')
    
    return f"{t1}{t2}{t3}"