# **TOYS Amazon Reviews BERT Fine-Tuning**  

Este projeto realiza o ajuste fino de um modelo BERT no [Conjunto de Dados de Avaliações de Clientes da Amazon EUA](https://www.kaggle.com/datasets/cynthiarempel/amazon-us-customer-reviews-dataset). Ele permite a análise de sentimentos e a classificação de avaliações.

---

## **Início Rápido 🚀**  

### **1 Instale as Dependências**  
```bash
pip install -r requirements.txt
```

### **2 Baixe as Dependências do Spark**
**Linux**
```bash
apt-get install openjdk-11-jdk-headless -qq > /dev/null
wget -q https://archive.apache.org/dist/spark/spark-3.5.2/spark-3.5.2-bin-hadoop3.tgz
tar xf spark-3.5.2-bin-hadoop3.tgz
pip -q install findspark
```

### **3 Baixe o Conjunto de Dados**  
```bash
python get_dataset.py
```

### **4 Ajuste Fino do Modelo BERT**  
```bash
python fine_tuning.py
```

---

## **Resumo do Projeto**  

- **📌 Objetivo:** Análise de Sentimentos & Classificação de Texto  
- **🤖 Modelo:** BERT (Ajustado)  
- **📊 Conjunto de Dados:** [Avaliações de Clientes da Amazon EUA](https://www.kaggle.com/datasets/cynthiarempel/amazon-us-customer-reviews-dataset) -> A TABELA UTILIZADA É "TOYS REVIEWS"  

---

## **Contribua**  

- **🐛 Encontrou um erro?** Envie uma issue!  
- **🚀 Quer melhorar este projeto?** Faça um fork e crie um pull request.  

---

## **Adicional**  

- **Dashboard:** Código do dashboard  
- **Medallion:** Código da Arquitetura Medallion  
- **Modelo de Inferência:** Código de Inferência  

---

## **Contato**  

Dúvidas? Entre em contato: **[andrecastrocq@gmail.com]**  

---

Esta versão inclui marcações adequadas em Markdown e mantém um estilo limpo e orientado à ação. Avise-me se precisar de seções adicionais! 🚀