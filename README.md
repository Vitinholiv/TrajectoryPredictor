# TrajectoryPredictor

Projeto que visa criar predições de trajetórias de um objeto em um determinado ambiente bidimensional. Esse projeto foi feito para o curso de Machine Learning para Equações Diferenciais do PICME.

## Instalação

Para instalar e começar a usar o projeto:
- Crie uma pasta no seu dispositivo e baixe os arquivos ou use o comando `git clone https://github.com/VitinDenoyr/TrajectoryPredictor.git`
- Instale o **Python** (de preferência, na versão 3.12) e as bibliotecas **NumPy**, **PyTorch**, **MatPlotLib**, **ImageIo** e **PyGame**. Você pode usar o comando `python -m pip install -r requirements.txt`.

## Uso

- Execute o **main.py** para um terminal interativo de opções que você pode usar.

<br/>

## Estrutura de Pastas

O projeto segue uma estrutura modular e orientada a objetos.

```
trajectory_project/
├── main.py                     # Ponto de entrada
├── pyproject.toml              # Configuração de pacote e dependências oficiais
├── configs/
│   └── default.py              # Dicionários de hiperparâmetros padrão (Default, Dummy)
├── runs/                       # Instâncias salvas (.pth e .json)
├── res/                        # Recursos visuais
└── src/
    ├── core/                   # Arquivos centrais do projeto
    │   ├── predictor.py        # Classe de estado e salvamento
    │   ├── trainer.py          # Responsável pelo treinamento
    │   └── losses.py           # Funções de perda
    ├── models/
    │   └── architectures.py    # Definição das Redes Neurais
    └── utils/                  # Ferramentas auxiliares
        ├── physics.py          # Integradores numéricos e física
        ├── visualizer.py       # Visualização de gráficos
        ├── simulator.py        # Simulador interativo
        └── helpers.py          # Funções auxiliares
```

# Previsão de Trajetória com Corpo Gravitacional usando Redes Neurais

Esse projeto tem o objetivo de criar um modelo de redes neurais capaz de, dado um cenário bidimensional com:
- Um corpo gravitacional em uma posição ($\alpha$,$\beta$) (m,m) e massa $M$ (kg);
- Uma posição objetivo ($x$,$y$) (m,m) no espaço;

Queremos prever valores de:
- Velocidade ($v$) (m/s);
- Direção ($\theta$) (rad);
- Tempo ($t$) (s);

De modo que ao lançar um projétil:
- Partindo da posição ($0$,$0$);
- Com um ângulo $\theta$;
- Com velocidade inicial $v$;

O projétil alcançará a posição ($x$,$y$) no tempo $t$, mesmo sob influência da gravidade do corpo gravitacional.

## Arquiteturas da Rede Neural

A arquitetura modela como serão as camadas, nós, entrada e saída da rede neural. Além disso, a arquitetura também é responsável por como os valores serão interpretados como velocidade, tempo e ângulo. 

**Observação**: $N$ é apenas um número que representa quantas observações foram passadas ao nó em um momento qualquer.
- SingleAim: Considera corpos gravitacionais como hiperparâmetros, tentando prever as direções apenas com base na posição do alvo.
  - Input: Tensor de tamanho ($N$,2), esperando as $N$ coordenadas ($x_i$,$y_i$) de cada alvo.
  - Output: Tensor de tamanho ($N$,3), retornando três valores que serão as estimativas de ($v$,$\theta$,$t$) para acertar o alvo em $t$ segundos.
  - Hidden Layers: $|Layers|$ camadas lineares com $Layers_i$ nós, cada uma seguida de uma função de ativação $Activations_i$.
  - Hiperparâmetros: Intervalos considerados para velocidade (m/s), tempo (s) e ângulo (rad), valores de $Layers$ e as funções de ativação $Activations$.

## Funções de Perda

As funções de perda são funções capazes de associar um valor numérico a seu par (input, output), representando quão errada a trajetória obtida pela saída da sua rede neural foi em relação ao cenário descrito pela entrada.
**Perdas Posicionais**: Funções de perda que penalizam a distância entre a posição do alvo e a posição final do projétil utilizando algum tipo de simulação numérica para estimar a posição do projétil no tempo dado.
- **loss_euler**: Utiliza o [Método de Euler Semi-Implícito](https://en.wikipedia.org/wiki/Semi-implicit_Euler_method) para iterar o movimento do projétil o projétil até a posição final.
- **loss_runge_kutta**: Utiliza o [Método de Runge-Kutta de Quarta Ordem](https://pt.wikipedia.org/wiki/M%C3%A9todo_de_Runge-Kutta) para iterar o movimento do projétil até a posição final. Usado para obter uma acurácia significativamente mais alta que o método de Euler semi-implícito. Apesar de ser mais caro para calcular, o ganho em acurácia reduz o número de passos necessários para uma precisão aceitável, tornando-o mais rápido especialmente em cenários que o treino está sendo realizado com uma GPU, reduzindo o bottleneck.

## Classe do Preditor

A classe do preditor se chama TrajectoryPredictor, e foi criada para lidar com todas as arquiteturas. Na inicialização, passamos uma string com o nome da arquitetura usada, uma string com qual função de perda será usada e um dicionário representando os hiperparâmetros usados no modelo e seus valores adotados. Mais a frente veremos quais os hiperparâmetros disponíveis e quais os valores padrão usados.

Os métodos da classe são os seguites:
- **_ _ init _ _**: Cria uma instância zerada do preditor, inicializando a arquitetura e guardando os hiperparâmetros.

- **save**: Salva o estado atual do preditor em um arquivo.

- **load**: Carrega um estado do preditor de um arquivo para o código.

- **calculate_loss**: Gera pontos aleatórios ao redor dos corpos gravitacionais dentro da distância definida pelo hiperparâmetro, utiliza os pesos atuais da rede neural para simular lançamentos e calcula a perda do modelo segundo a função dada.

- **fit**: Executa o treinamento do modelo com as configurações especificadas.

- **plot_loss**: Exibe um gráfico mostrando a relação entre a perda do modelo em função das épocas.

- **predict**: Executa e plota predições de pontos específicos ou de pontos aleatórios usando o modelo treinado atual.

- **simulate**: Inicia uma simulação do lançamento de projéteis com PyGame. Usando o mouse, é possível controlar a direção e velocidade do projétil para o lançamento. Também é possível ver e utilizar a direção e velocidade previstas pelo modelo atual para uma dada configuração do cenário.

## Configurações do Preditor

Agora que vimos as partes que compõem o projeto, podemos ver como são definidas novas configurações. Eles são criados a partir de um modelo padrão **defaultModel**, que possui valores padrão. A partir disso, podemos criar um novo modelo usando a função **create_model** fornecendo um id e um dicionário de modificações nos hiperparâmetros que representam a nova configuração.
- **defaultModel**: Um modelo genérico que define um estado padrão simples de ser executado. Modelo com um treino rápido utilizando a CPU e possuindo apenas um corpo gravitacional, sendo tratado como hiperparâmetro.
- **dummyModel**: Uma versão simplificada aleatória com um número ínfimo de épocas. Não é nada eficiente, servindo apenas como placeholder para entender como criar novas configurações e também sendo uma versão para executar de forma muito rápida uma etapa de treinamento, mesmo sem obter bons resultados.

## Treinamento

Para executar o treinamento, usamos o método **fit** da classe do modelo. Abaixo, criamos uma instância e a treinamos (se já não estiver treinada).

## Utilitários



