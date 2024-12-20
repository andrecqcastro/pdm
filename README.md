# **TOYS Amazon Reviews BERT Fine-Tuning**  

This project fine-tunes a BERT model on the [Amazon US Customer Reviews Dataset](https://www.kaggle.com/datasets/cynthiarempel/amazon-us-customer-reviews-dataset). It enables sentiment analysis and review classification.

---

## **Quick Start 🚀**  

### **1 Install Dependencies**  
```bash
pip install -r requirements.txt
```

### **2 Download Spark Dependencies**
**Linux**
```bash
apt-get install openjdk-11-jdk-headless -qq > /dev/null
wget -q https://archive.apache.org/dist/spark/spark-3.5.2/spark-3.5.2-bin-hadoop3.tgz
tar xf spark-3.5.2-bin-hadoop3.tgz
pip -q install findspark
```

### **3 Download  the Dataset**  
```bash
python get_dataset.py
```

### **4 Fine-Tune the BERT Model**  
```bash
python fine_tuning.py
```

---

## **Project Summary**  

- **📌 Goal:** Sentiment Analysis & Text Classification  
- **🤖 Model:** BERT (Fine-Tuned)  
- **📊 Dataset:** [Amazon US Customer Reviews](https://www.kaggle.com/datasets/cynthiarempel/amazon-us-customer-reviews-dataset) -> TOYS REVIEWS IS THE TABLE WE ARE USING 

---

## **Contribute**  

- **🐛 Found a bug?** Submit an issue!  
- **🚀 Want to improve this project?** Fork it and create a pull request.  

---

## **Adicional**

- **Dashboard:** dashboard code
- **Medallion:** Medallion Architecture code
- **Inference Model:** Inferece code.


## **Contact**  

Questions? Reach out at **[andrecastrocq@gmail.com]**.  

---

This version includes proper Markdown marks and keeps a clean, action-driven style. Let me know if you need additional sections! 🚀


