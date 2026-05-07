# Guarda funções relacionadas a física
import torch
import scipy.constants as const

def loss_euler_step(bX: torch.Tensor, bY:torch.Tensor, bMass:torch.Tensor, xi:torch.Tensor, yi:torch.Tensor,
                    vx: torch.Tensor, vy:torch.Tensor, bias:float, dt:torch.Tensor):
    """
    Calcula o próximo estado (posição e velocidade) do projétil após um intervalo de tempo `dt`
    usando o método de integração numérica de Euler (Forward Euler).

    Args:
        bX (torch.Tensor): Posições X dos corpos gravitacionais.
        bY (torch.Tensor): Posições Y dos corpos gravitacionais.
        bMass (torch.Tensor): Massas dos corpos gravitacionais.
        xi (torch.Tensor): Posição X atual do projétil.
        yi (torch.Tensor): Posição Y atual do projétil.
        vx (torch.Tensor): Velocidade em X atual do projétil.
        vy (torch.Tensor): Velocidade em Y atual do projétil.
        bias (float): Fator de suavização (Plummer Softening) para evitar singularidades (divisão por zero).
        dt (torch.Tensor): Tamanho do passo de tempo (Time step).

    Returns:
        tuple: (xi, yi, vx, vy) atualizados para o próximo passo de tempo.
    """

    rVec_x = bX.view(1, -1) - xi
    rVec_y = bY.view(1, -1) - yi
    dist_sq = rVec_x**2 + rVec_y**2
    dist_cubed = (dist_sq + bias**2)**1.5

    ax = torch.sum((const.G * bMass.view(1, -1) * rVec_x)/dist_cubed, dim=1, keepdim=True)
    ay = torch.sum((const.G * bMass.view(1, -1) * rVec_y)/dist_cubed, dim=1, keepdim=True)
    
    vx = vx + ax * dt
    vy = vy + ay * dt
    xi = xi + vx * dt
    yi = yi + vy * dt
    return xi,yi,vx,vy

def loss_runge_kutta_slope(bX: torch.Tensor, bY:torch.Tensor, bMass:torch.Tensor, xi:torch.Tensor, yi:torch.Tensor,
                           bias:float):
    """
    Calcula a aceleração gravitacional (inclinação/derivada da velocidade) exercida pelos corpos
    sobre o projétil em uma dada posição (xi, yi). Função auxiliar para o método de Runge-Kutta.

    Args:
        bX (torch.Tensor): Posições X dos corpos gravitacionais.
        bY (torch.Tensor): Posições Y dos corpos gravitacionais.
        bMass (torch.Tensor): Massas dos corpos gravitacionais.
        xi (torch.Tensor): Posição X do projétil para cálculo.
        yi (torch.Tensor): Posição Y do projétil para cálculo.
        bias (float): Fator de suavização (Plummer Softening).

    Returns:
        tuple: (ax, ay) representando as acelerações nos eixos X e Y.
    """

    rVec_x = bX.view(1, -1) - xi
    rVec_y = bY.view(1, -1) - yi
    
    dist_sq = rVec_x**2 + rVec_y**2
    dist_cubed = (dist_sq + bias**2)**1.5 # Plummer Softening

    ax = torch.sum((const.G * bMass.view(1, -1) * rVec_x) / dist_cubed, dim=1, keepdim=True)
    ay = torch.sum((const.G * bMass.view(1, -1) * rVec_y) / dist_cubed, dim=1, keepdim=True)
    return ax, ay

def loss_runge_kutta_step(bX: torch.Tensor, bY: torch.Tensor, bMass: torch.Tensor, xi: torch.Tensor, yi: torch.Tensor,
                          vx: torch.Tensor, vy: torch.Tensor, bias:float, dt:torch.Tensor):
    """
    Calcula o próximo estado (posição e velocidade) do projétil após um intervalo de tempo `dt`
    usando o método de integração numérica de Runge-Kutta de 4ª Ordem (RK4).

    Args:
        bX (torch.Tensor): Posições X dos corpos gravitacionais.
        bY (torch.Tensor): Posições Y dos corpos gravitacionais.
        bMass (torch.Tensor): Massas dos corpos gravitacionais.
        xi (torch.Tensor): Posição X atual do projétil.
        yi (torch.Tensor): Posição Y atual do projétil.
        vx (torch.Tensor): Velocidade em X atual do projétil.
        vy (torch.Tensor): Velocidade em Y atual do projétil.
        bias (float): Fator de suavização (Plummer Softening).
        dt (torch.Tensor): Tamanho do passo de tempo (Time step).

    Returns:
        tuple: (xi, yi, vx, vy) atualizados pelo método RK4.
    """

    ax1, ay1 = loss_runge_kutta_slope(bX, bY, bMass, xi, yi, bias)
    kvx1, kvy1 = ax1, ay1
    kx1, ky1   = vx, vy

    ax2, ay2 = loss_runge_kutta_slope(bX, bY, bMass, xi + 0.5 * dt * kx1, yi + 0.5 * dt * ky1, bias)
    kvx2, kvy2 = ax2, ay2
    kx2, ky2   = vx + 0.5 * dt * kvx1, vy + 0.5 * dt * kvy1

    ax3, ay3 = loss_runge_kutta_slope(bX, bY, bMass, xi + 0.5 * dt * kx2, yi + 0.5 * dt * ky2, bias)
    kvx3, kvy3 = ax3, ay3
    kx3, ky3   = vx + 0.5 * dt * kvx2, vy + 0.5 * dt * kvy2

    ax4, ay4 = loss_runge_kutta_slope(bX, bY, bMass, xi + dt * kx3, yi + dt * ky3, bias)
    kvx4, kvy4 = ax4, ay4
    kx4, ky4   = vx + dt * kvx3, vy + dt * kvy3

    xi = xi + (dt / 6.0) * (kx1 + 2*kx2 + 2*kx3 + kx4)
    yi = yi + (dt / 6.0) * (ky1 + 2*ky2 + 2*ky3 + ky4)
    vx = vx + (dt / 6.0) * (kvx1 + 2*kvx2 + 2*kvx3 + kvx4)
    vy = vy + (dt / 6.0) * (kvy1 + 2*kvy2 + 2*kvy3 + kvy4)

    return xi, yi, vx, vy