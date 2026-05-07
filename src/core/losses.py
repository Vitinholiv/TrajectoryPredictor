# Guarda as funções de perda
import torch

from src.utils.physics import loss_euler_step, loss_runge_kutta_step

def loss_euler(v: torch.Tensor, th: torch.Tensor, t: torch.Tensor, x: torch.Tensor, y: torch.Tensor, bX: torch.Tensor,
               bY: torch.Tensor, bMass: torch.Tensor, bias: float, steps: int):
    """
    Calcula o Erro Quadrático Médio (MSE) do destino do projétil em relação ao alvo (x,y),
    utilizando a integração numérica de Euler. Esta função é otimizada para o treinamento da rede.

    Args:
        v (torch.Tensor): Velocidade inicial prevista pela rede.
        th (torch.Tensor): Ângulo de lançamento (em radianos) previsto pela rede.
        t (torch.Tensor): Tempo total de voo previsto pela rede.
        x (torch.Tensor): Coordenada X do alvo real.
        y (torch.Tensor): Coordenada Y do alvo real.
        bX (torch.Tensor): Coordenadas X dos corpos gravitacionais.
        bY (torch.Tensor): Coordenadas Y dos corpos gravitacionais.
        bMass (torch.Tensor): Massas dos corpos gravitacionais.
        bias (float): Fator de suavização (Plummer Softening) para evitar singularidades gravitacionais.
        steps (int): Número de passos (iterações) para a integração numérica do tempo 't'.

    Returns:
        torch.Tensor: Um valor escalar contendo o Erro Quadrático Médio (MSE) de todo o batch.
    """
    
    xi = torch.zeros_like(x)
    yi = torch.zeros_like(y)
    dt = t / steps
    vx = v * torch.cos(th)
    vy = v * torch.sin(th)

    for _ in range(steps):
        xi, yi, vx, vy = loss_euler_step(bX, bY, bMass, xi, yi, vx, vy, bias, dt)

    loss = (xi - x)**2 + (yi - y)**2
    return torch.mean(loss)

def loss_euler_trajectory(v: torch.Tensor, th: torch.Tensor, t: torch.Tensor, x: torch.Tensor, y: torch.Tensor,
                          bX: torch.Tensor, bY: torch.Tensor, bMass: torch.Tensor, bias: float, steps: int):
    """
    Calcula o erro quadrático do destino do projétil em relação ao alvo (x,y) via método de Euler, 
    retornando também a trajetória completa percorrida passo a passo. Utilizada para visualização e métricas.

    Args:
        v, th, t, x, y, bX, bY, bMass, bias, steps: Mesmos parâmetros da função `loss_euler`.

    Returns:
        tuple: Contém três elementos:
            - xCoords (numpy.ndarray): Histórico completo das posições X do projétil ao longo do tempo.
            - yCoords (numpy.ndarray): Histórico completo das posições Y do projétil ao longo do tempo.
            - loss (torch.Tensor): O erro quadrático de cada amostra.
    """

    xi = torch.zeros_like(x)
    yi = torch.zeros_like(y)
    dt = t / steps
    vx = v * torch.cos(th)
    vy = v * torch.sin(th)
    
    xCoords = [xi.clone()]
    yCoords = [yi.clone()]

    for _ in range(steps):
        xi, yi, vx, vy = loss_euler_step(bX, bY, bMass, xi, yi, vx, vy, bias, dt)
        xCoords.append(xi.clone())
        yCoords.append(yi.clone())

    loss = (xi - x)**2 + (yi - y)**2
    return torch.stack(xCoords).detach().numpy(), torch.stack(yCoords).detach().numpy(), loss.detach().cpu()

def loss_runge_kutta(v: torch.Tensor, th: torch.Tensor, t: torch.Tensor, x: torch.Tensor, y: torch.Tensor,
                     bX: torch.Tensor, bY: torch.Tensor, bMass: torch.Tensor, bias: float, steps: int):
    """
    Calcula o Erro Quadrático Médio (MSE) do destino do projétil em relação ao alvo (x,y),
    utilizando a integração numérica de Runge-Kutta de 4ª Ordem (RK4). Otimizada para treinamento.

    Args:
        v (torch.Tensor): Velocidade inicial prevista pela rede.
        th (torch.Tensor): Ângulo de lançamento (em radianos) previsto pela rede.
        t (torch.Tensor): Tempo total de voo previsto pela rede.
        x (torch.Tensor): Coordenada X do alvo real.
        y (torch.Tensor): Coordenada Y do alvo real.
        bX (torch.Tensor): Coordenadas X dos corpos gravitacionais.
        bY (torch.Tensor): Coordenadas Y dos corpos gravitacionais.
        bMass (torch.Tensor): Massas dos corpos gravitacionais.
        bias (float): Fator de suavização (Plummer Softening).
        steps (int): Número de passos de integração.

    Returns:
        torch.Tensor: Um valor escalar contendo o Erro Quadrático Médio (MSE) do batch.
    """
    xi = torch.zeros_like(x)
    yi = torch.zeros_like(y)
    dt = t / steps
    
    vx = v * torch.cos(th)
    vy = v * torch.sin(th)

    for _ in range(steps):
        xi, yi, vx, vy = loss_runge_kutta_step(bX, bY, bMass, xi, yi, vx, vy, bias, dt)

    loss = (xi - x)**2 + (yi - y)**2
    return torch.mean(loss)

def loss_runge_kutta_trajectory(v: torch.Tensor, th: torch.Tensor, t: torch.Tensor, x: torch.Tensor, y: torch.Tensor,
                                bX: torch.Tensor, bY: torch.Tensor, bMass: torch.Tensor, bias: float, steps: int):
    """
    Calcula o erro quadrático do projétil via método Runge-Kutta (RK4), retornando também 
    a trajetória percorrida passo a passo. Devido à precisão do RK4, esta trajetória 
    oferece a melhor aproximação física do voo orbital para plotagens.

    Args:
        v, th, t, x, y, bX, bY, bMass, bias, steps: Mesmos parâmetros da função `loss_runge_kutta`.

    Returns:
        tuple: Contém três elementos:
            - xCoords (numpy.ndarray): Histórico completo das posições X da trajetória.
            - yCoords (numpy.ndarray): Histórico completo das posições Y da trajetória.
            - loss (torch.Tensor): O erro quadrático de cada amostra.
    """
    xi = torch.zeros_like(x)
    yi = torch.zeros_like(y)
    dt = t / steps
    
    vx = v * torch.cos(th)
    vy = v * torch.sin(th)

    xCoords = [xi.clone()]
    yCoords = [yi.clone()]

    for _ in range(steps):
        xi, yi, vx, vy = loss_runge_kutta_step(bX, bY, bMass, xi, yi, vx, vy, bias, dt)
        xCoords.append(xi.clone())
        yCoords.append(yi.clone())

    loss = (xi - x)**2 + (yi - y)**2
    return torch.stack(xCoords).detach().numpy(), torch.stack(yCoords).detach().numpy(), loss.detach().cpu()