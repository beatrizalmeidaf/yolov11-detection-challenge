# Avaliação de Modelos YOLOv11 com Diferentes Resoluções para Detecção de Veículos

Esse repositório apresenta a solução para o desafio técnico de estágio em Machine Learning, com foco em detecção de objetos utilizando o framework **YOLOv11 (Ultralytics)**.

O relatório completo, contendo todos os resultados e análises detalhadas, está disponível no link abaixo:

📄 [Relatório Final - PDF](https://github.com/beatrizalmeidaf/yolov11-detection-challenge/blob/main/relatorio-beatrizalmeida-desafio-disbral.pdf)

> **Observação:** Os dados utilizados no projeto foram removidos desse repositório para garantir a anonimização. Caso deseje acessar o notebook diretamente no Google Colab, utilize o link abaixo:
>
> 🔗 [Notebook no Google Colab](https://drive.google.com/file/d/1iTwIhn1we2A7IEs5S9foHsu7ohFl99Bw/view?usp=sharing)

---

## Objetivo do Projeto

Treinar três modelos de detecção de objetos com diferentes resoluções de imagem de entrada (`imgsz`), mantendo todos os demais hiperparâmetros constantes, e compará-los por meio de métricas consistentes de avaliação.

Modelos:

- **Modelo A:** imgsz = 256x256  
- **Modelo B:** imgsz = 512x512  
- **Modelo C:** imgsz = 640x640  

Todos os modelos foram treinados por 50 épocas com pesos pré-treinados.

---

## Estratégia de Treinamento

- **Framework:** Ultralytics YOLOv11  
- **Épocas:** 50  
- **Dataset:** Fornecido pelo desafio (divisão 80/20)  
- **Batch size, otimizador e taxa de aprendizado:** constantes  
- **Execução:** Ambiente Google Colab com integração ao Google Drive  

---

## Métricas de Avaliação

As métricas utilizadas para avaliação dos modelos foram:

- **F1-score**
- **Precision**
- **Recall**
- **mAP@50**
- **mAP@50-95**

Essas métricas foram escolhidas para permitir uma avaliação tanto quantitativa quanto qualitativa da performance dos modelos.

---

## Resultados Resumidos

| Modelo | imgsz | F1-score | Precision | Recall | mAP@50 | mAP@50-95 |
|--------|-------|----------|-----------|--------|--------|------------|
| A      | 256   | 0.6468   | 0.7951    | 0.5451 | 0.5515 | 0.4088     |
| B      | 512   | 0.7572   | 0.8419    | 0.6880 | 0.7432 | 0.6511     |
| C      | 640   | 0.7572   | 0.9173    | 0.6447 | 0.7480 | 0.6880     |

**Resumo das análises:**

- O modelo com **512px** apresentou o melhor equilíbrio entre desempenho e custo computacional.
- O modelo com **640px** alcançou o melhor resultado em mAP@50-95, indicando maior generalização.
- O modelo com **256px** teve desempenho inferior, mas com menor custo de processamento.

---

## Como Executar o Projeto

1. Clone o repositório:
   ```bash
   git clone https://github.com/beatrizalmeidaf/yolov11-detection-challenge.git
   cd yolov11-detection-challenge
   ```

2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

3. Abra o Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

4. Execute o notebook `YOLOv11_Detection_Challenge.ipynb` para visualizar o pipeline completo de pré-processamento, treinamento e avaliação.

> **Importante:** Os dados utilizados nesse projeto foram removidos para garantir anonimização.  
> Para executar o notebook com seus próprios dados, insira os arquivos na pasta apropriada do repositório, seguindo a mesma estrutura esperada pelo script. Certifique-se de ajustar os caminhos no notebook, se necessário.

---

## Considerações Finais

O projeto foi conduzido com foco em organização, reprodutibilidade e análise crítica dos resultados. A avaliação comparativa entre os modelos com diferentes resoluções de entrada permitiu observar o impacto do tamanho da imagem na capacidade de detecção e generalização.

Para detalhes adicionais, consulte o relatório completo em PDF disponível neste repositório.
