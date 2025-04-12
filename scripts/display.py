import os
import yaml
import cv2
import random
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def display_dataset_info():
    """
    Exibe informações sobre a estrutura do dataset, incluindo a contagem de imagens 
    e rótulos nos diretórios de treino e validação. Também calcula e mostra a 
    proporção de divisão (split) entre treino e validação.
    
    Retorna:
        dict: Contagem de imagens e rótulos para treino e validação.
    """
    
    base_dir = Path('./data_train/dataset')
    train_images_dir = base_dir / 'train' / 'images'
    val_images_dir = base_dir / 'val' / 'images'
    
    train_labels_dir = base_dir / 'train' / 'labels'
    val_labels_dir = base_dir / 'val' / 'labels'
    
    # contagem de arquivos
    train_images_count = len(list(train_images_dir.glob('*'))) if train_images_dir.exists() else 0
    val_images_count = len(list(val_images_dir.glob('*'))) if val_images_dir.exists() else 0
    train_labels_count = len(list(train_labels_dir.glob('*'))) if train_labels_dir.exists() else 0
    val_labels_count = len(list(val_labels_dir.glob('*'))) if val_labels_dir.exists() else 0
    
    print(f"Estrutura do Dataset:")
    print(f"- Train images: {train_images_count}")
    print(f"- Train labels: {train_labels_count}")
    print(f"- Validation images: {val_images_count}")
    print(f"- Validation labels: {val_labels_count}")
    
    # verificação da proporção de split
    total_images = train_images_count + val_images_count
    train_percentage = (train_images_count / total_images) * 100 if total_images > 0 else 0
    val_percentage = (val_images_count / total_images) * 100 if total_images > 0 else 0
    
    print(f"\nProporções de Split:")
    print(f"- Train: {train_percentage:.1f}% ({train_images_count} images)")
    print(f"- Validation: {val_percentage:.1f}% ({val_images_count} images)")
    
    return {
        'train_images': train_images_count,
        'val_images': val_images_count,
        'train_labels': train_labels_count,
        'val_labels': val_labels_count
    }

def read_yaml_config(yaml_path='./yaml/data.yaml'):
    """
    Lê e interpreta um arquivo de configuração YAML. 
    Garante que o campo 'names', se presente como dicionário, seja convertido para uma lista ordenada.
    
    Parâmetros:
        yaml_path (str): Caminho para o arquivo YAML.
        
    Retorna:
        dict | None: Dicionário com os dados do YAML, ou None se o arquivo não for encontrado.
    """
    
    if not os.path.exists(yaml_path):
        print(f"Arquivo YAML não encontrado em {yaml_path}")
        return None 

    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)

    # garante que 'names' serão tratados como uma lista
    if isinstance(config['names'], dict):
        names_list = [config['names'][i] for i in sorted(config['names'])]
        config['names'] = names_list  # substitui por lista

    return config


def visualize_dataset_samples(num_samples=2):
    """
    Visualiza amostras aleatórias do conjunto de treino, desenhando caixas delimitadoras 
    com base nos rótulos disponíveis. Cada imagem é exibida com as anotações de classe (se disponíveis).
    
    Parâmetros:
        num_samples (int): Número de amostras a serem exibidas (padrão: 3).
    """
    try:
        # carrega as configurações do YAML
        with open('./yaml/data.yaml', 'r') as f:
            yaml_config = yaml.safe_load(f)
    except Exception as e:
        print(f"Erro ao carregar o arquivo data.yaml: {e}")
        return

    train_images_dir = Path('./data_train/dataset/train/images')
    train_labels_dir = Path('./data_train/dataset/train/labels')
    
    if not train_images_dir.exists() or not train_labels_dir.exists():
        print(f"Diretórios não encontrados:\n - Imagens: {train_images_dir}\n - Rótulos: {train_labels_dir}")
        return

    # extensões válidas para imagem
    valid_exts = ['.jpg', '.jpeg', '.png']
    image_files = [f for f in train_images_dir.glob('*') if f.suffix.lower() in valid_exts]
    
    if not image_files:
        print("Nenhuma imagem válida encontrada para visualização.")
        return

    sample_files = random.sample(image_files, min(num_samples, len(image_files)))

    fig, axes = plt.subplots(1, len(sample_files), figsize=(5 * len(sample_files), 5))
    if len(sample_files) == 1:
        axes = [axes]

    for i, img_path in enumerate(sample_files):
        img = cv2.imread(str(img_path))
        
        if img is None:
            print(f"Erro ao carregar a imagem: {img_path}")
            axes[i].set_title('Erro ao carregar imagem')
            axes[i].axis('off')
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, _ = img.shape

        label_path = train_labels_dir / (img_path.stem + '.txt')

        if label_path.exists():
            try:
                with open(label_path, 'r') as f:
                    lines = f.readlines()

                for line in lines:
                    data = line.strip().split()
                    if len(data) >= 5:
                        class_id = int(data[0])
                        x_center, y_center, box_width, box_height = map(float, data[1:5])

                        # coordenadas absolutas
                        x1 = int((x_center - box_width / 2) * width)
                        y1 = int((y_center - box_height / 2) * height)
                        x2 = int((x_center + box_width / 2) * width)
                        y2 = int((y_center + box_height / 2) * height)

                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

                        if 'names' in yaml_config and class_id < len(yaml_config['names']):
                            class_name = yaml_config['names'][class_id]
                            cv2.putText(img, class_name, (x1, max(15, y1 - 5)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            except Exception as e:
                print(f"Erro ao processar rótulo para {img_path.name}: {e}")

        axes[i].imshow(img)
        axes[i].set_title(f'Image: {img_path.name}')
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


def analyze_class_distribution():
    """
    Analisa e visualiza a distribuição total de classes no dataset.
    Mostra:
    - Gráfico de barras agrupadas (Train vs Val)
    - Gráfico horizontal com a proporção total por classe
    """
    # carrega as configurações do YAML
    with open('./yaml/data.yaml', 'r') as f:
        yaml_config = yaml.safe_load(f)

    # converte o dict de classes para lista ordenada
    class_names = [name for _, name in sorted(yaml_config['names'].items())]

    # diretórios dos labels
    train_labels_dir = Path('./data_train/dataset/train/labels')
    val_labels_dir = Path('./data_train/dataset/val/labels')

    # inicializa contadores
    class_counts_train = {name: 0 for name in class_names}
    class_counts_val = {name: 0 for name in class_names}

    # conta classes no treino
    if train_labels_dir.exists():
        for label_file in train_labels_dir.glob('*.txt'):
            with open(label_file, 'r') as f:
                for line in f:
                    class_id = int(line.strip().split()[0])
                    if 0 <= class_id < len(class_names):
                        class_name = class_names[class_id]
                        class_counts_train[class_name] += 1
                    else:
                        print(f"[AVISO] ID de classe inválido ({class_id}) encontrado em {label_file}")

    # conta classes na validação
    if val_labels_dir.exists():
        for label_file in val_labels_dir.glob('*.txt'):
            with open(label_file, 'r') as f:
                for line in f:
                    class_id = int(line.strip().split()[0])
                    if 0 <= class_id < len(class_names):
                        class_name = class_names[class_id]
                        class_counts_val[class_name] += 1
                    else:
                        print(f"[AVISO] ID de classe inválido ({class_id}) encontrado em {label_file}")

    # criação do DataFrame
    df = pd.DataFrame({
        'Class': list(class_counts_train.keys()),
        'Train': list(class_counts_train.values()),
        'Validation': list(class_counts_val.values())
    })
    df['Total'] = df['Train'] + df['Validation']
    df['% Total'] = (df['Total'] / df['Total'].sum()) * 100
    df = df.sort_values(by='Total', ascending=True)

    # plotagem
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    # gráfico de barras agrupadas
    df.plot(x='Class', y=['Train', 'Validation'], kind='bar', ax=ax1, color=['#4C72B0', '#55A868'])
    ax1.set_title('Distribuição de Classes por Conjunto', fontsize=14, weight='bold')
    ax1.set_ylabel('Ocorrências')
    ax1.set_xlabel('Classes')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', linestyle='--', alpha=0.6)
    for bars in ax1.containers:
        ax1.bar_label(bars, fmt='%d', label_type='edge', fontsize=9)

    # gráfico de proporções
    ax2.barh(df['Class'], df['% Total'], color='#1f77b4')
    ax2.set_title('Proporção Total por Classe', fontsize=14, weight='bold')
    ax2.set_xlabel('% do Dataset')
    ax2.grid(axis='x', linestyle='--', alpha=0.6)

    for i, (percent, total) in enumerate(zip(df['% Total'], df['Total'])):
        ax2.text(percent + 0.5, i, f"{percent:.1f}% ({total})", va='center', fontsize=9)

    plt.tight_layout()
    plt.show()

    return df