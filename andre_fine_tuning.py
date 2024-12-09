import pandas as pd
import gzip
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, mean_absolute_error, roc_auc_score
import matplotlib.pyplot as plt
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
import torch
import numpy as np
import os
import findspark
from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"  # ATUALIZAR PARA O DIRETÓRIO DO SEU COMPUTADOR
os.environ["SPARK_HOME"] = "/home/pressprexx/Software/spark-3.5.2-bin-hadoop3"

# Garantindo que o Spark seja configurado corretamente
findspark.init()

spark = SparkSession.builder.appName('BERT Fine Tuning').master("local[*]").getOrCreate()

# Caminho do arquivo
file_path = './selected_data/amazon_reviews_us_Toys_v1_00.tsv'

df = spark.read.csv(file_path, header=True, sep='\t', inferSchema=True)

# Selecione as colunas relevantes
df_selected = df.select("review_body", "star_rating")

# Remova linhas com valores nulos
df_clean = df_selected.dropna(subset=["review_body", "star_rating"])
print(f"Quantidade total de linhas após limpeza: {df_clean.count()}")


# Transforme as classes
df_transformed = df_clean.withColumn(
    "star_rating",
    F.when(F.col("star_rating").isin(1, 2), 1)   # Negativo
     .when(F.col("star_rating") == 3, 2)         # Regular
     .when(F.col("star_rating").isin(4, 5), 3)   # Positivo
)

# Verifique o balanceamento
class_frequency = df_transformed.groupBy("star_rating").count().orderBy("star_rating")

# Exiba os resultados
print("Frequência de cada classe antes do balanceamento:")
class_frequency.show()

# Defina a quantidade desejada
quantidade_desejada = 387668

# Balanceie as classes usando limit
df_class_1 = df_transformed.filter(F.col("star_rating") == 1).limit(quantidade_desejada)
df_class_2 = df_transformed.filter(F.col("star_rating") == 2).limit(quantidade_desejada)
df_class_3 = df_transformed.filter(F.col("star_rating") == 3).limit(quantidade_desejada)

# Combine as classes balanceadas
df_balanced = df_class_1.union(df_class_2).union(df_class_3)

# Shuffle o DataFrame
df_balanced = df_balanced.orderBy(rand())

# Verifique o balanceamento
class_frequency_balanced = df_balanced.groupBy("star_rating").count().orderBy("star_rating")

# Exiba os resultados
print("Frequência de cada classe após balanceamento:")
class_frequency_balanced.show()

# Quantidade total de linhas após o balanceamento
total_linhas = df_balanced.count()

# Calcula as proporções
train_size = int(total_linhas * 0.7)
test_size = int(total_linhas * 0.2)
val_size = total_linhas - train_size - test_size

# Divide o DataFrame em partes
df_train = df_balanced.limit(train_size)

# Remove as linhas de treino para gerar o DataFrame de teste
df_restante = df_balanced.subtract(df_train)
df_test = df_restante.limit(test_size)

# Gera o DataFrame de validação com o restante
df_val = df_restante.subtract(df_test)

# Verifique o balanceamento

class_frequency = df_train.groupBy("star_rating").count().orderBy("star_rating")
print("Frequência de cada classe no df_train:")
class_frequency.show()


class_frequency = df_test.groupBy("star_rating").count().orderBy("star_rating")
print("Frequência de cada classe no df_test:")
class_frequency.show()

class_frequency = df_val.groupBy("star_rating").count().orderBy("star_rating")
print("Frequência de cada classe no df_val:")
class_frequency.show()


# Verifique as contagens
print(f"Linhas de treinamento: {df_train.count()}")
print(f"Linhas de teste: {df_test.count()}")
print(f"Linhas de validação: {df_val.count()}")


tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

def tokenize(batch):
    tokenized_inputs = tokenizer(batch['review_body'], padding=True, truncation=True, max_length=128, return_tensors='pt')
    tokenized_inputs["labels"] = torch.tensor(batch['star_rating'])
    return tokenized_inputs

train_dataset = Dataset.from_spark(df_train).map(tokenize, batched=True)
test_dataset = Dataset.from_spark(df_test).map(tokenize, batched=True)

train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

# Initializing the model
model = AutoModelForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=3
)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    learning_rate=1e-5,
    weight_decay=0.01,
    logging_dir='./logs',
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_steps=10,
    fp16=True
)

# Function to compute metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions

    # Hard predictions are needed for accuracy, precision, recall, and F1
    hard_preds = np.argmax(preds, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, hard_preds, average='weighted')
    acc = accuracy_score(labels, hard_preds)
    mae = mean_absolute_error(labels, hard_preds)

    # Compute ROC AUC for each class
    roc_auc = {}
    for i in range(preds.shape[1]):  # Iterate over each class
        roc_auc[f"roc_auc_class_{i}"] = roc_auc_score((labels == i).astype(int), preds[:, i])

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'mae': mae,
        **roc_auc  # This will expand the dictionary to include the roc_auc for each class
    }


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset.map(tokenize, batched=True),
    eval_dataset=test_dataset.map(tokenize, batched=True),
    compute_metrics=compute_metrics,
)

# Training the model
trainer.train()

# Evaluating the model on the test dataset
trainer.evaluate()