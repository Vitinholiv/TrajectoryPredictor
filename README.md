# TrajectoryPredictor

Projeto que visa criar predições de trajetórias de um objeto em um determinado ambiente bidimensional. Esse projeto foi feito para o curso de Machine Learning para Equações Diferenciais do PICME.

## Instalação

Para instalar e começar a usar o projeto:
- Crie uma pasta no seu dispositivo e baixe os arquivos ou use o comando `git clone https://github.com/VitinDenoyr/TrajectoryPredictor.git`
- Instale o **Python** (de preferência, na versão 3.12) e as bibliotecas **NumPy**, **PyTorch**, **MatPlotLib**, **ImageIo** e **PyGame**. Você pode usar o comando `python -m pip install -r requirements.txt`.

## Uso

- Dentro da pasta base, encontre o arquivo do notebook **notebook.ipynb** para ter uma visão geral do código completo do projeto. Ele inclui:
   - Classes que descrevem as possíveis arquiteturas da rede neural.
   - Funções de Perda implementadas.
   - Classe do modelo com estrutura de inicialização e salvamento de instâncias.
   - Métodos adicionais para realizar treinamento, previsão e simulação.

- Ainda na pasta base, existe o arquivo **train.py**, sendo um script contendo uma versão reduzida do código feito para treinar novos modelos.