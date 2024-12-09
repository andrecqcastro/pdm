import kagglehub
import os
import shutil

# Download o dataset completo
path = kagglehub.dataset_download("cynthiarempel/amazon-us-customer-reviews-dataset")

# Defina o nome do arquivo desejado
desired_file = 'amazon_reviews_us_Toys_v1_00.tsv'
source_file = os.path.join(path, desired_file)

# Verifique se o arquivo existe
if os.path.exists(source_file):
    # Mova ou copie para um diretório específico
    target_dir = './selected_data'
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(source_file, target_dir)
    print(f"Arquivo '{desired_file}' copiado para '{target_dir}'.")
else:
    print(f"Arquivo '{desired_file}' não encontrado.")
