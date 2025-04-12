# Desafio Técnico Para Estágio em Machine Learning 🚀

Esse repositório contém a minha solução para o desafio técnico voltado para estágio em Machine Learning. O objetivo é treinar e analisar modelos de detecção de objetos utilizando o framework **YOLOv11**, com foco em diferentes configurações de pré-processamento de imagens.

---

## Contexto 📌

A tarefa proposta simula o desafio enfrentado por uma empresa fictícia que atua na avaliação de obras de infraestrutura de transportes. A ideia é desenvolver uma IA capaz de realizar **contagem volumétrica de veículos** em obras, auxiliando engenheiros nas decisões sobre melhorias e dimensionamento de pavimentos.

---

## Objetivo 🎯

- Treinar modelos de detecção com o **YOLOv11** utilizando imagens com diferentes resoluções:  
  - **Modelo A:** 256x256  
  - **Modelo B:** 512x512  
  - **Modelo C:** 640x640  
- Manter todos os outros parâmetros de treinamento constantes (épocas, batch size, otimizador, etc.)
- Utilizar modelo pré-treinado com pesos atualizados durante o treinamento
- Rodar cada modelo por pelo menos **50 épocas**
- Comparar os modelos utilizando **três métricas de avaliação**

---

## Métricas Escolhidas 

As métricas selecionadas para análise comparativa dos modelos foram:

1. **F1 Score:**  
   Combina precisão e revocação em uma única métrica, ideal para avaliar equilíbrio entre falsos positivos e falsos negativos, especialmente importante em problemas de detecção.

2. **Matriz de Confusão:**  
   Ajuda a identificar como o modelo está classificando cada classe e onde ocorrem os maiores erros.

3. **PR Curve (Precision-Recall Curve):**  
   Permite visualizar a troca entre precisão e revocação ao longo de diferentes thresholds de confiança, útil para entender melhor a performance dos modelos com múltiplas classes.

Essas métricas foram escolhidas por oferecerem uma visão mais **qualitativa e interpretável** da performance dos modelos além dos números brutos.

---

## Como Rodar o Projeto 

1. Clone este repositório:
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

4. Execute o notebook `YOLOv11_Detection_Challenge.ipynb`, que contém todo o código de pré-processamento, treinamento e análise comparativa dos modelos.

---

## Status do Projeto ⚠️

Infelizmente, **não consegui concluir os treinamentos nem gerar os resultados** devido a **limitações de hardware** no meu ambiente local. Durante as execuções, o consumo de memória RAM foi muito alto, resultando em **crashes constantes do kernel**.

Apesar disso, toda a estrutura do projeto foi implementada, incluindo:
- Configuração do ambiente e dependências
- Treinamento estruturado para os três modelos
- Códigos prontos para gerar as análises comparativas

Assim que eu tiver acesso a um ambiente com mais recursos, pretendo retomar e finalizar os experimentos.

---

## Considerações Finais 

Mesmo sem conseguir finalizar os treinamentos por limitações técnicas, me preocupei em estruturar o projeto de forma clara e organizada, seguindo boas práticas de codificação, documentação e análise.

Agradeço pela oportunidade de participar do desafio! 

Se houver qualquer dúvida ou sugestão, estou à disposição para conversar ou adaptar o projeto.
