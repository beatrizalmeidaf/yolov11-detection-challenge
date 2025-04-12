import yaml
import os
import time
import shutil
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
import torch
import traceback

def create_model_configs(base_yaml_path='./yaml/data.yaml'):
    """Cria arquivos de configuração YAML para diferentes tamanhos de imagem"""
    configs = []
    image_sizes = [256, 512, 640]
    
    for img_size in image_sizes:
        config_path = f'./yaml/data_{img_size}.yaml'
        
        with open(base_yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # adiciona o tamanho da imagem ao arquivo de configuração
        config['path'] = 'data_train/dataset'
        config['imgsz'] = img_size
        
        # salva o novo arquivo YAML
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        configs.append(config_path)
    
    return configs

def train_model(img_size, epochs=50, batch_size=16, model_path='yolo11n.pt', force_cpu=True):
    """Treina um modelo YOLO com os parâmetros especificados e salva métricas e resultados"""
    model = None
    save_dir = None
    
    try:
        # desativa o clearML para evitar problemas de conexão
        os.environ["CLEARML_AGENT_SKIP_PYTHON_ENV_INSTALL"] = "1"
        os.environ["CLEARML_OFF"] = "1"

        os.environ["TORCH_COMPILE"] = "0"
        torch._dynamo.config.suppress_errors = True
        
        # extrai o nome do modelo para nomear diretórios
        model_name = os.path.basename(model_path).split('.')[0]
        
        # cria diretório de saída baseado no timestamp e tamanho da imagem
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = f"/content/yolo_outputs/train_{model_name}_imgsz{img_size}_{timestamp}"
        
        # garante que o diretório base existe
        os.makedirs("/content/yolo_outputs", exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"iniciando treinamento do modelo {model_name} com imgsz={img_size}")
        print(f"{'='*80}")
        
        # copia o arquivo do modelo para armazenamento local se estiver no google drive
        local_model_path = f"/content/{os.path.basename(model_path)}"
        if model_path.startswith('/content/drive'):
            print(f"copiando modelo de {model_path} para {local_model_path}...")
            shutil.copy(model_path, local_model_path)
            model_path = local_model_path
            
        if not os.path.exists(model_path):
            print(f"erro: arquivo {model_path} não encontrado!")
            return None, None
        
        # define o dispositivo forçando cpu se necessário
        device = 'cpu' if force_cpu else ('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"usando dispositivo: {device}")
        
        # copia o dataset para o armazenamento local para evitar problemas de conexão com o drive
        original_dataset_path = "/content/drive/MyDrive/ml_challenge/yolo_project/data_train/dataset"
        local_dataset_path = "/content/dataset"
        
        if not os.path.exists(local_dataset_path):
            print(f"copiando dataset de {original_dataset_path} para {local_dataset_path}...")
            shutil.copytree(original_dataset_path, local_dataset_path)
            
        dataset_path = local_dataset_path
        
        train_path = os.path.join(dataset_path, "train")
        val_path = os.path.join(dataset_path, "val")
        
        if not os.path.exists(train_path):
            print(f"erro: diretório de treino não encontrado: {train_path}")
            return None, None
            
        if not os.path.exists(val_path):
            print(f"erro: diretório de validação não encontrado: {val_path}")
            return None, None
            
        print(f"diretório de treino: {train_path}")
        print(f"diretório de validação: {val_path}")
        
        # cria arquivo yaml temporário para configuração do dataset 
        yaml_path = "/content/dataset_config.yaml"
        with open(yaml_path, 'w') as f:
            yaml_content = {
                "path": dataset_path,
                "train": "train/images",
                "val": "val/images",
                "nc": 6,
                "names": ["person", "car", "motorcycle", "bus", "truck", "van"]
            }
            yaml.dump(yaml_content, f, default_flow_style=False)
        
        start_time = time.time()
        
        print(f"usando dataset em: {dataset_path}")
        print(f"iniciando treinamento...")

        # carrega o modelo
        model = YOLO(model_path)

        # inicia o treinamento com a configuração atualizada
        results = model.train(
            data=yaml_path,
            imgsz=img_size,
            epochs=epochs,
            batch=batch_size,
            name=os.path.basename(save_dir),
            project="/content/yolo_outputs",
            patience=10,
            device=device,
            workers=0,  
            save_period=10,
            exist_ok=True, optimizer='AdamW',
            lr0=0.001
        )

        # calcula o tempo de treinamento
        training_time = time.time() - start_time
        hours, remainder = divmod(training_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        print(f"\ntreinamento concluído em {int(hours)}h {int(minutes)}m {int(seconds)}s")
        print(f"resultados salvos em: {model.trainer.save_dir}")
        
        save_dir = model.trainer.save_dir
        
        # salva os resultados com o tamanho da imagem claramente identificado
        results_csv = os.path.join(save_dir, "results.csv")
        results_summary_path = "" 
        
        if os.path.exists(results_csv):
            # lê o csv completo
            df = pd.read_csv(results_csv)
            
            # adiciona uma coluna para o tamanho da imagem
            df['img_size'] = img_size
            
            # extrai apenas os resultados da última época para resumo
            last_epoch_data = df.iloc[-1:].copy()
            
            # salva o arquivo de resumo com o tamanho da imagem no nome
            results_summary_path = os.path.join(save_dir, f"results_summary_{img_size}px.csv")
            last_epoch_data.to_csv(results_summary_path, index=False)
            
            print(f"resumo da última época salvo em: {results_summary_path}")
            
            # renomeia o arquivo de resultados original para incluir o tamanho da imagem
            renamed_results_csv = os.path.join(save_dir, f"results_{img_size}px.csv")
            shutil.copy(results_csv, renamed_results_csv)
        
        # validação opcional do modelo após o treinamento
        final_metrics = {}
        metrics_summary_path = ""  
        
        try:
            if hasattr(model.trainer, 'best') and os.path.exists(model.trainer.best):
                print(f"\navaliando modelo com melhor desempenho: {model.trainer.best}")
                val_results = model.val(data=yaml_path)
                
                map50_95 = val_results.box.map
                map50 = val_results.box.map50
                
                print(f"mAP50-95: {map50_95}")
                print(f"mAP50: {map50}")
                
                # extrai precision e recall do último checkpoint
                if os.path.exists(results_csv):
                    df = pd.read_csv(results_csv)
                    if 'metrics/precision(B)' in df.columns and 'metrics/recall(B)' in df.columns:
                        precision = df['metrics/precision(B)'].iloc[-1]
                        recall = df['metrics/recall(B)'].iloc[-1]
                        
                        # calcula f1-score
                        if precision + recall > 0:
                            f1_score = 2 * (precision * recall) / (precision + recall)
                        else:
                            f1_score = 0.0
                        
                        print(f"precision: {precision:.4f}")
                        print(f"recall: {recall:.4f}")
                        print(f"f1-score: {f1_score:.4f}")
                        
                        # guarda as métricas finais
                        final_metrics = {
                            'img_size': img_size,
                            'map50': map50,
                            'map50_95': map50_95,
                            'precision': precision,
                            'recall': recall,
                            'f1_score': f1_score
                        }
                        
                        # salva métricas finais em arquivo separado
                        metrics_summary_path = os.path.join(save_dir, f"metrics_summary_{img_size}px.csv")
                        pd.DataFrame([final_metrics]).to_csv(metrics_summary_path, index=False)
                        print(f"métricas finais salvas em: {metrics_summary_path}")
        except Exception as validation_e:
            print(f"erro durante a validação: {validation_e}")

        # copia os resultados de volta para o google drive se necessário
        try:
            drive_results_dir = "/content/drive/MyDrive/ml_challenge/yolo_project/results"
            # cria uma pasta específica para o tamanho da imagem
            img_size_dir = os.path.join(drive_results_dir, f"{img_size}px")
            os.makedirs(img_size_dir, exist_ok=True)
            
            # copia o melhor modelo e os resultados principais
            if hasattr(model.trainer, 'best') and os.path.exists(model.trainer.best):
                best_model_filename = f"{img_size}px_{timestamp}_best.pt"
                drive_best_path = os.path.join(img_size_dir, best_model_filename)
                shutil.copy(model.trainer.best, drive_best_path)
                print(f"melhor modelo copiado para: {drive_best_path}")
                
            if os.path.exists(renamed_results_csv):
                drive_csv_path = os.path.join(img_size_dir, f"{img_size}px_{timestamp}_results.csv")
                shutil.copy(renamed_results_csv, drive_csv_path)
                print(f"resultados completos copiados para: {drive_csv_path}")
            
            if os.path.exists(results_summary_path):
                drive_summary_path = os.path.join(img_size_dir, f"{img_size}px_{timestamp}_results_summary.csv")
                shutil.copy(results_summary_path, drive_summary_path)
                print(f"resumo da última época copiado para: {drive_summary_path}")
            
            if os.path.exists(metrics_summary_path):
                drive_metrics_path = os.path.join(img_size_dir, f"{img_size}px_{timestamp}_metrics_summary.csv")
                shutil.copy(metrics_summary_path, drive_metrics_path)
                print(f"métricas finais copiadas para: {drive_metrics_path}")
            
            # atualiza um arquivo consolidado com todas as métricas no diretório principal
            consolidated_path = os.path.join(drive_results_dir, "consolidated_metrics.csv")
            if os.path.exists(consolidated_path):
                consolidated_df = pd.read_csv(consolidated_path)
                if final_metrics:
                    new_row = pd.DataFrame([{
                        'timestamp': timestamp,
                        'img_size': img_size,
                        'model_name': model_name,
                        'map50': final_metrics.get('map50', 0),
                        'map50_95': final_metrics.get('map50_95', 0),
                        'precision': final_metrics.get('precision', 0),
                        'recall': final_metrics.get('recall', 0),
                        'f1_score': final_metrics.get('f1_score', 0)
                    }])
                    consolidated_df = pd.concat([consolidated_df, new_row], ignore_index=True)
                    consolidated_df.to_csv(consolidated_path, index=False)
                    print(f"métricas adicionadas ao consolidado: {consolidated_path}")
            else:
                # cria um novo arquivo consolidado se não existir
                if final_metrics:
                    new_df = pd.DataFrame([{
                        'timestamp': timestamp,
                        'img_size': img_size,
                        'model_name': model_name,
                        'map50': final_metrics.get('map50', 0),
                        'map50_95': final_metrics.get('map50_95', 0),
                        'precision': final_metrics.get('precision', 0),
                        'recall': final_metrics.get('recall', 0),
                        'f1_score': final_metrics.get('f1_score', 0)
                    }])
                    new_df.to_csv(consolidated_path, index=False)
                    print(f"novo arquivo consolidado criado em: {consolidated_path}")
        except Exception as copy_e:
            print(f"aviso: erro ao copiar arquivos para o drive: {copy_e}")
        
        print(f"Treinamento para imagem de tamanho {img_size}px finalizado com sucesso!")
        return model, save_dir
    
    except Exception as e:
        print(f"erro durante o treinamento: {e}")
        traceback.print_exc()
        return None, None
    
    # retorno de segurança caso alguma exceção não tratada ocorra
    return model, save_dir
