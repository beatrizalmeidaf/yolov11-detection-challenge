import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# configuração global para todos os gráficos
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'figure.figsize': (12, 8),
    'figure.dpi': 100,
    'axes.grid': True,
    'grid.alpha': 0.3
})

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

def analyze_yolo_models(base_dir='/content/drive/MyDrive/ml_challenge/yolo_project/results/'):
    """
    Analisa automaticamente todos os modelos YOLO encontrados no diretório especificado,
    comparando-os pelo tamanho de imagem usando o arquivo consolidado de métricas.
    
    Args:
        base_dir: Diretório base onde estão os resultados dos modelos
    """
    print(f"Buscando modelos no diretório: {base_dir}")
    
    # procura o arquivo consolidado de métricas
    consolidated_path = os.path.join(base_dir, "consolidated_metrics.csv")
    
    if os.path.exists(consolidated_path):
        print(f"Arquivo consolidado de métricas encontrado: {consolidated_path}")
        # analisa usando o arquivo consolidado 
        analyze_from_consolidated(consolidated_path)
    else:
        print("Arquivo consolidado de métricas não encontrado. Executando análise de arquivos individuais...")
        # continua com a análise baseada em arquivos individuais
        analyze_from_individual_files(base_dir)

def analyze_from_consolidated(consolidated_path):
    """
    Analisa e visualiza métricas usando o arquivo consolidado.
    """
    try:
        df = pd.read_csv(consolidated_path)
        
        if df.empty:
            print("Arquivo consolidado de métricas está vazio.")
            return
        
        print(f"Carregadas {len(df)} linhas de dados do arquivo consolidado.")
        
        
        # agrupa por tamanho de imagem e modelo
        if 'img_size' in df.columns and 'model_name' in df.columns:
            # cria uma coluna de identificação única combinando tamanho de imagem e modelo
            df['model_id'] = df['img_size'].astype(str) + 'px_' + df['model_name']
            
            # para cada modelo_id obtém a linha com o melhor F1-score
            best_models = df.loc[df.groupby('model_id')['f1_score'].idxmax()]
            
            # agrupa apenas por tamanho de imagem 
            metrics_by_size = best_models[['img_size', 'model_name', 'precision', 'recall', 'f1_score', 'map50', 'map50_95']]

            print("\nMétricas dos melhores modelos por tamanho de imagem:")
            print(metrics_by_size)

            # plota as métricas por tamanho de imagem
            plot_consolidated_metrics(metrics_by_size)
            
            # encontra o melhor tamanho de imagem baseado no F1-score
            best_size_idx = metrics_by_size['f1_score'].idxmax()
            best_size = metrics_by_size.iloc[best_size_idx]
            
            print(f"\nMelhor tamanho de imagem: {int(best_size['img_size'])}px")
            print(f"F1-score: {best_size['f1_score']:.4f}")
            print(f"Precision: {best_size['precision']:.4f}")
            print(f"Recall: {best_size['recall']:.4f}")
            print(f"mAP50: {best_size['map50']:.4f}")
            print(f"mAP50-95: {best_size['map50_95']:.4f}")
            
        else:
            print("Colunas necessárias não encontradas no arquivo consolidado.")
    except Exception as e:
        print(f"Erro ao analisar arquivo consolidado: {str(e)}")
        import traceback
        traceback.print_exc()

def plot_consolidated_metrics(metrics_df):
    """
    Plota gráficos comparativos a partir do DataFrame de métricas consolidadas.
    """
    # converte o tamanho de imagem para string com "px" para melhor visualização
    metrics_df['img_size_str'] = metrics_df['img_size'].astype(str) + 'px'
    metrics_df = metrics_df.sort_values('img_size') 

    with plt.style.context('seaborn-v0_8-whitegrid'):
        # 1. Gráfico de barras para F1-score
        fig, ax = plt.subplots(figsize=(12, 7))
        
        bars = ax.bar(
            metrics_df['img_size_str'], 
            metrics_df['f1_score'], 
            color=COLORS[0],
            width=0.6,
            edgecolor='black',
            linewidth=1.5,
            alpha=0.8
        )
        
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2., 
                height + 0.01,
                f'{height:.3f}', 
                ha='center', 
                va='bottom',
                fontweight='bold'
            )

        ax.set_title('Comparação de F1-Score por Tamanho de Imagem', fontsize=18, pad=20)
        ax.set_xlabel('Tamanho de Imagem', fontsize=14, labelpad=10)
        ax.set_ylabel('F1-Score', fontsize=14, labelpad=10)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.set_ylim(0, min(1.0, metrics_df['f1_score'].max() * 1.2))
        
        best_idx = metrics_df['f1_score'].idxmax()
        best_size = metrics_df.loc[best_idx, 'img_size_str']
        ax.get_xticklabels()[list(metrics_df['img_size_str']).index(best_size)].set_weight('bold')
        ax.get_xticklabels()[list(metrics_df['img_size_str']).index(best_size)].set_color('darkred')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
        
        plt.tight_layout()
        plt.show()
        
        # 2. Gráfico comparativo de todas as métricas
        # dados para gráfico comparativo
        plot_data = metrics_df.melt(
            id_vars=['img_size_str'],
            value_vars=['precision', 'recall', 'f1_score', 'map50', 'map50_95'],
            var_name='Métrica',
            value_name='Valor'
        )
        
        metric_names = {
            'precision': 'Precisão', 
            'recall': 'Recall', 
            'f1_score': 'F1 Score',
            'map50': 'mAP50', 
            'map50_95': 'mAP50-95'
        }
        plot_data['Métrica'] = plot_data['Métrica'].map(metric_names)
        
        metric_order = ['Precisão', 'Recall', 'F1 Score', 'mAP50', 'mAP50-95']
        plot_data['Métrica'] = pd.Categorical(
            plot_data['Métrica'], 
            categories=metric_order, 
            ordered=True
        )
        
        plt.figure(figsize=(14, 9))
        
        ax = sns.barplot(
            x='img_size_str', 
            y='Valor', 
            hue='Métrica', 
            data=plot_data,
            palette=COLORS[:5],
            edgecolor='black',
            linewidth=1.2,
            alpha=0.8
        )

        for c in ax.containers:
            labels = [f'{v:.2f}' if v >= 0.2 else '' for v in c.datavalues]
            ax.bar_label(c, labels=labels, padding=3, fontsize=10)
        
        plt.title('Comparação de Métricas por Tamanho de Imagem', fontsize=18, pad=20)
        plt.xlabel('Tamanho de Imagem', fontsize=14, labelpad=10)
        plt.ylabel('Score', fontsize=14, labelpad=10)
        plt.legend(title='Métrica', title_fontsize=12, fontsize=11, loc='upper left', frameon=True)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.ylim(0, 1.05)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.show()
        
        # 3. Radar chart para visualização integrada das métricas
        plot_radar_chart(metrics_df)

def plot_radar_chart(metrics_df):
    """
    Cria um gráfico radar (teia de aranha) para visualizar todas as métricas
    por tamanho de imagem.
    """
    metrics = ['precision', 'recall', 'f1_score', 'map50', 'map50_95']
    N = len(metrics)
    
    # angulos para o gráfico radar (divide o círculo em N partes iguais)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # fecha o círculo

    cmap = plt.cm.viridis
    custom_colors = [cmap(i) for i in np.linspace(0, 0.8, len(metrics_df))]
    
    metrics_df = metrics_df.sort_values('img_size')
    
    fig = plt.figure(figsize=(12, 10), facecolor='white')
    ax = fig.add_subplot(111, polar=True)
    
    metric_labels = ['Precisão', 'Recall', 'F1 Score', 'mAP50', 'mAP50-95']
    
    # configura os rótulos
    plt.xticks(angles[:-1], metric_labels, size=14)
    
    for i in range(1, 6):
        level = i/5.0
        ax.plot(angles, [level]*len(angles), '--', color='gray', alpha=0.3)
        if i < 5: 
            plt.text(np.pi/10, level, f'{level:.1f}', ha='center', va='center', size=10, color='gray')
    
    # plota cada tamanho de imagem
    for i, row in metrics_df.iterrows():
        values = [row[metric] for metric in metrics]
        values += values[:1]  # fecha o círculo
        
        ax.plot(
            angles, 
            values, 
            linewidth=2.5, 
            label=f"{int(row['img_size'])}px", 
            color=custom_colors[i],
            alpha=0.8
        )
        ax.fill(angles, values, alpha=0.15, color=custom_colors[i])

    ax.set_facecolor('#f8f9fa')
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    ax.legend(
        loc='upper right', 
        bbox_to_anchor=(0.15, 0.15),
        frameon=True,
        title='Tamanho da Imagem',
        title_fontsize=14,
        fontsize=12
    )
    
    plt.title(
        'Comparação de Métricas por Tamanho de Imagem', 
        fontsize=18, 
        pad=20,
        fontweight='bold'
    )
    
    plt.tight_layout()
    plt.show()

def analyze_from_individual_files(base_dir):
    """
    Realiza análise baseada nos arquivos individuais de resultados.
    Usa o mesmo código da função original analyze_yolo_models.
    """
    # encontra todos os arquivos CSV de resultados
    csv_files = glob.glob(os.path.join(base_dir, '*', '*results.csv'))
    csv_files.sort() 
    
    if not csv_files:
        print("Nenhum arquivo de resultados CSV encontrado.")
        return
    
    print(f"Encontrados {len(csv_files)} arquivos de resultados.")
    
    # extrai informações de cada modelo
    models_info = []
    
    for csv_file in csv_files:
        base_name = os.path.basename(csv_file).split('_results.csv')[0]
        parent_dir = os.path.dirname(csv_file)
        
        # procura pelo modelo best.pt correspondente
        model_path = os.path.join(parent_dir, f"{base_name}_best.pt")

        try:
            # tenta extrair do nome do diretório pai
            parent_dirname = os.path.basename(parent_dir)
            if 'px' in parent_dirname:
                img_size = int(parent_dirname.split('px')[0])
            else:
                # tenta extrair de subdiretórios com padrão de tamanho
                for size in ['256', '512', '640']:
                    if size in parent_dirname:
                        img_size = int(size)
                        break
                else:
                    df = pd.read_csv(csv_file)
                    if 'img_size' in df.columns:
                        img_size = df['img_size'].iloc[0]
                    else:
                        img_size = "Desconhecido"
        except:
            img_size = "Desconhecido"
        
        models_info.append({
            'csv_file': csv_file,
            'model_path': model_path,
            'base_name': base_name,
            'parent_dir': parent_dir,
            'img_size': img_size
        })
    
    # análise de métricas a partir dos CSVs agrupando por tamanho de imagem
    analyze_metrics_by_image_size(models_info)
    
    # procura arquivos de visualização
    find_and_display_visualizations(models_info)

def analyze_metrics_by_image_size(models_info):
    """
    Função original para analisar métricas por tamanho de imagem a partir de arquivos individuais.
    """
    # filtra apenas modelos com tamanho de imagem conhecido e numérico
    valid_models = [model for model in models_info if isinstance(model['img_size'], (int, float))]
    
    if not valid_models:
        print("Nenhum modelo com tamanho de imagem válido encontrado.")
        return
    
    # agrupa modelos por tamanho de imagem
    models_by_size = {}
    for model in valid_models:
        size = model['img_size']
        if size not in models_by_size:
            models_by_size[size] = []
        models_by_size[size].append(model)
    
    final_metrics = {
        'precision': [],
        'recall': [],
        'f1_score': [],
        'mAP50': [],
        'mAP50-95': [],
        'img_sizes': []
    }

    for img_size, models in models_by_size.items():
        print(f"\nAnalisando modelos com tamanho de imagem: {img_size}px")
        
        # para cada tamanho pega as métricas do melhor modelo (baseado em F1)
        best_f1 = -1
        best_metrics = None
        
        for model in models:
            if os.path.exists(model['csv_file']):
                df = pd.read_csv(model['csv_file'])
                
                # verifica se possui as métricas necessárias
                if 'metrics/precision(B)' in df.columns and 'metrics/recall(B)' in df.columns:
                    # pega os valores da última época 
                    final_precision = df['metrics/precision(B)'].iloc[-1]
                    final_recall = df['metrics/recall(B)'].iloc[-1]
                    
                    # calcula F1 score
                    if final_precision + final_recall > 0:
                        final_f1 = 2 * (final_precision * final_recall) / (final_precision + final_recall)
                    else:
                        final_f1 = 0.0
                  
                    if final_f1 > best_f1:
                        best_f1 = final_f1
                        
                        # extrai métricas mAP
                        map50 = df['metrics/mAP50(B)'].iloc[-1] if 'metrics/mAP50(B)' in df.columns else 0
                        map50_95 = df['metrics/mAP50-95(B)'].iloc[-1] if 'metrics/mAP50-95(B)' in df.columns else 0
                        
                        best_metrics = {
                            'precision': final_precision,
                            'recall': final_recall,
                            'f1_score': final_f1,
                            'mAP50': map50,
                            'mAP50-95': map50_95
                        }

        if best_metrics:
            print(f"  Melhor F1-score para modelo {img_size}px: {best_metrics['f1_score']:.4f}")
            final_metrics['precision'].append(best_metrics['precision'])
            final_metrics['recall'].append(best_metrics['recall'])
            final_metrics['f1_score'].append(best_metrics['f1_score'])
            final_metrics['mAP50'].append(best_metrics['mAP50'])
            final_metrics['mAP50-95'].append(best_metrics['mAP50-95'])
            final_metrics['img_sizes'].append(img_size)
    
    if final_metrics['img_sizes']:
        # ordena métricas por tamanho de imagem
        sorted_indices = np.argsort(final_metrics['img_sizes'])
        for key in final_metrics:
            final_metrics[key] = [final_metrics[key][i] for i in sorted_indices]

        img_sizes_str = [f"{size}px" for size in final_metrics['img_sizes']]
        
        with plt.style.context('seaborn-v0_8-whitegrid'):
            # 1. Gráfico comparativo de F1-score
            fig, ax = plt.subplots(figsize=(12, 7))
            
            bars = ax.bar(
                img_sizes_str, 
                final_metrics['f1_score'],
                color=COLORS[0],
                width=0.6,
                edgecolor='black',
                linewidth=1.5,
                alpha=0.8
            )
            
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2., 
                    height + 0.01,
                    f'{height:.3f}', 
                    ha='center', 
                    va='bottom',
                    fontweight='bold'
                )
            
            ax.set_title('Comparação de F1-Score por Tamanho de Imagem', fontsize=18, pad=20)
            ax.set_xlabel('Tamanho de Imagem', fontsize=14, labelpad=10)
            ax.set_ylabel('F1-Score', fontsize=14, labelpad=10)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            ax.set_ylim(0, min(1.0, max(final_metrics['f1_score']) * 1.15))
            
            best_idx = final_metrics['f1_score'].index(max(final_metrics['f1_score']))
            ax.get_xticklabels()[best_idx].set_weight('bold')
            ax.get_xticklabels()[best_idx].set_color('darkred')

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            plt.tight_layout()
            plt.show()
            
            # 2. Gráfico comparativo de todas as métricas
            fig, ax = plt.subplots(figsize=(14, 8))
            x = np.arange(len(img_sizes_str))
            width = 0.17
            
            metric_colors = COLORS[:5]
            
            # barra para cada métrica
            rects1 = ax.bar(x - width*2, final_metrics['precision'], width, label='Precisão', 
                            color=metric_colors[0], edgecolor='black', linewidth=1, alpha=0.8)
            rects2 = ax.bar(x - width, final_metrics['recall'], width, label='Recall', 
                            color=metric_colors[1], edgecolor='black', linewidth=1, alpha=0.8)
            rects3 = ax.bar(x, final_metrics['f1_score'], width, label='F1 Score', 
                            color=metric_colors[2], edgecolor='black', linewidth=1, alpha=0.8)
            rects4 = ax.bar(x + width, final_metrics['mAP50'], width, label='mAP50', 
                            color=metric_colors[3], edgecolor='black', linewidth=1, alpha=0.8)
            rects5 = ax.bar(x + width*2, final_metrics['mAP50-95'], width, label='mAP50-95', 
                            color=metric_colors[4], edgecolor='black', linewidth=1, alpha=0.8)
            
            # adicionar valores nas barras
            def autolabel(rects):
                for rect in rects:
                    height = rect.get_height()
                    if height >= 0.2: 
                        ax.annotate(f'{height:.2f}',
                                    xy=(rect.get_x() + rect.get_width() / 2, height),
                                    xytext=(0, 3),  
                                    textcoords="offset points",
                                    ha='center', va='bottom',
                                    fontsize=9)
            
            autolabel(rects1)
            autolabel(rects2)
            autolabel(rects3)
            autolabel(rects4)
            autolabel(rects5)
            
            ax.set_ylabel('Score', fontsize=14)
            ax.set_xlabel('Tamanho de Imagem', fontsize=14)
            ax.set_title('Comparação de Métricas por Tamanho de Imagem', fontsize=18, pad=20)
            ax.set_xticks(x)
            ax.set_xticklabels(img_sizes_str)
            ax.legend(fontsize=12, frameon=True)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            ax.set_ylim(0, 1.05)
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            plt.tight_layout()
            plt.show()
            
            # 3. Comparação de mAP
            fig, ax = plt.subplots(figsize=(12, 7))
            width = 0.35
            
            bar1 = ax.bar(x - width/2, final_metrics['mAP50'], width, label='mAP50', 
                          color=metric_colors[3], edgecolor='black', linewidth=1, alpha=0.8)
            bar2 = ax.bar(x + width/2, final_metrics['mAP50-95'], width, label='mAP50-95', 
                          color=metric_colors[4], edgecolor='black', linewidth=1, alpha=0.8)

            autolabel(bar1)
            autolabel(bar2)
            
            ax.set_xlabel('Tamanho de Imagem', fontsize=14, labelpad=10)
            ax.set_ylabel('Score', fontsize=14, labelpad=10)
            ax.set_title('Comparação de mAP por Tamanho de Imagem', fontsize=18, pad=20)
            ax.set_xticks(x)
            ax.set_xticklabels(img_sizes_str)
            ax.legend(fontsize=12, frameon=True)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            ax.set_ylim(0, min(1.05, max(max(final_metrics['mAP50']), max(final_metrics['mAP50-95'])) * 1.15))

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            plt.tight_layout()
            plt.show()
            
            # 4. Radar chart para melhor comparação entre tamanhos
            plot_individual_radar_chart(final_metrics)

def plot_individual_radar_chart(final_metrics):
    """
    Cria um gráfico radar para os dados individuais
    """
    metrics = ['precision', 'recall', 'f1_score', 'mAP50', 'mAP50-95']
    N = len(metrics)
    
    # angulos para o gráfico radar
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # fecha o círculo

    img_sizes = final_metrics['img_sizes']
    img_sizes_str = [f"{size}px" for size in img_sizes]
    
    cmap = plt.cm.viridis
    custom_colors = [cmap(i) for i in np.linspace(0, 0.8, len(img_sizes))]
    
    fig = plt.figure(figsize=(12, 10), facecolor='white')
    ax = fig.add_subplot(111, polar=True)
    
    metric_labels = ['Precisão', 'Recall', 'F1 Score', 'mAP50', 'mAP50-95']
    
    plt.xticks(angles[:-1], metric_labels, size=14)
    
    for i in range(1, 6):
        level = i/5.0
        ax.plot(angles, [level]*len(angles), '--', color='gray', alpha=0.3)
        if i < 5:  
            plt.text(np.pi/10, level, f'{level:.1f}', ha='center', va='center', size=10, color='gray')
    
    # plota cada tamanho de imagem
    for i, size in enumerate(img_sizes):
        values = [
            final_metrics['precision'][i],
            final_metrics['recall'][i],
            final_metrics['f1_score'][i],
            final_metrics['mAP50'][i],
            final_metrics['mAP50-95'][i]
        ]
        values += values[:1]  # fecha o círculo
        
        ax.plot(
            angles, 
            values, 
            linewidth=2.5, 
            label=f"{size}px", 
            color=custom_colors[i],
            alpha=0.8
        )
        ax.fill(angles, values, alpha=0.15, color=custom_colors[i])

    ax.set_facecolor('#f8f9fa')
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    ax.legend(
        loc='upper right', 
        bbox_to_anchor=(0.15, 0.15),
        frameon=True,
        title='Tamanho da Imagem',
        title_fontsize=14,
        fontsize=12
    )
    
    plt.title(
        'Comparação de Métricas por Tamanho de Imagem', 
        fontsize=18, 
        pad=20,
        fontweight='bold'
    )
    
    plt.tight_layout()
    plt.show()

def find_and_display_visualizations(models_info):
    """
    Procura e exibe visualizações como matrizes de confusão e curvas PR.
    """
    print("\n--- Procurando Visualizações ---")
    
    # agrupa modelos por tamanho de imagem para visualizações
    models_by_size = {}
    for model in models_info:
        size = model['img_size']
        if size not in models_by_size:
            models_by_size[size] = []
        models_by_size[size].append(model)
    
    # procura visualizações relevantes para cada tamanho de imagem
    for img_size, models in models_by_size.items():
        print(f"\nVisualizações para modelos com tamanho {img_size}px:")
        
        if models:
            model = models[0]  
            
            base_dir = model['parent_dir']
            
            search_paths = [
                base_dir, 
                os.path.join(base_dir, 'val'),
                os.path.join(base_dir, 'labels'), 
                os.path.join(base_dir, 'weights')  
            ]
            
            # tipos de visualizações para procurar
            viz_types = {
                'confusion_matrix': ['confusion_matrix.png', 'confusion*.png', '*confusion*.png'],
                'pr_curve': ['PR_curve.png', 'pr_curve*.png', '*pr*.png'],
                'f1_curve': ['F1_curve.png', 'f1_curve*.png', '*f1*.png']
            }
            
            found_any = False
            
            # layout mais organizado para múltiplas visualizações
            num_viz_types = len(viz_types)
            found_viz = {}
    
            for viz_name, patterns in viz_types.items():
                for search_path in search_paths:
                    if os.path.exists(search_path):
                        for pattern in patterns:
                            matches = glob.glob(os.path.join(search_path, pattern))
                            if matches:
                                found_viz[viz_name] = matches[0]
                                found_any = True
                                print(f"  Encontrado {viz_name}: {matches[0]}")
                                break
                    if viz_name in found_viz:
                        break
    
            if found_any:
                num_found = len(found_viz)
                if num_found == 1:
                    viz_name, viz_path = next(iter(found_viz.items()))
                    plt.figure(figsize=(12, 10))
                    img = plt.imread(viz_path)
                    plt.imshow(img)
                    plt.title(f'{viz_name.replace("_", " ").title()} - Modelo {img_size}px', fontsize=16)
                    plt.axis('off')
                    plt.tight_layout()
                    plt.show()
                else:
                    rows = (num_found + 1) // 2  
                    cols = min(2, num_found)
                    
                    fig, axes = plt.subplots(rows, cols, figsize=(14, 5*rows))
                    fig.suptitle(f'Visualizações para Modelo {img_size}px', fontsize=18, y=0.98)
                    
                    if num_found > 1:
                        ax_flat = axes.flatten() if rows > 1 else axes
                    else:
                        ax_flat = [axes]
                    
                    for i, (viz_name, viz_path) in enumerate(found_viz.items()):
                        if i < len(ax_flat):
                            img = plt.imread(viz_path)
                            ax_flat[i].imshow(img)
                            ax_flat[i].set_title(f'{viz_name.replace("_", " ").title()}', fontsize=14)
                            ax_flat[i].axis('off')

                    for i in range(num_found, len(ax_flat)):
                        ax_flat[i].axis('off')
                    
                    plt.tight_layout()
                    plt.subplots_adjust(top=0.92)
                    plt.show()
            else:
                print("  Nenhuma visualização encontrada para este tamanho de imagem.")