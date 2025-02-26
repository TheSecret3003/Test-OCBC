# User Classification and Financial Chatbot WebApp

This repository contains a web app developed using Streamlit and hosted on Streamlit Cloud. The projects covered are:

- User Classification
- Financial Chatbot 

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Dataset Description](#dataset-description)
5. [Technologies Used](#technologies-used)
6. [Model Development Process](#model-development-process)
7. [Models Used](#models-used)
8. [Model Evaluation](#model-evaluation)
9. [Conclusion](#conclusion)
10. [Deployment](#deployment)
11. [Contributing](#contributing)
12. [Contact](#contact)

## Overview

### User Classification
This web application allows users to classify user based on the input features. The classification model was developed through extensive data analysis and model selection processes, ensuring high accuracy and reliability.

### Financial Chatbot

This project implements a Retrieval-Augmented Generation (RAG) chatbot for financial statements using:

1. Streamlit for the frontend

2. ChromaDB for document retrieval

3. Lightweight open-source LLM (e.g., Gemma-2B or TinyLlama-1.1B) for response generation

4. LangChain for seamless integration of retrieval and generation

The chatbot allows users to query financial statements and receive answers supported by relevant sources (search-engine-like behavior).

## Installation

To run this project locally, please follow these steps:

1. Clone the repository
2. Navigate to the project directory
3. Install the required dependencies

```bash
git clone <repository_url>
cd <project_directory>
pip install -r requirements.txt
```

## Usage

To start the Streamlit web app, run the following command in your terminal:

```bash
streamlit run streamlit_app.py
```

This will launch the web app in your default web browser. You can then select the desired AI projects from the sidebar and input the required features to get a prediction.

## Dataset Description

### User Classification
**Description:** The data is a user level data containing all demographic-related data, Facebook data and several phone-specific data (aggregated to user level).All variables with prefix: 'de' comes from user's self reported data, prefix: 'fb' from user's Facebook profile and 'ph' from user's phone data.

### Financial Chatbot
**Description:** pdf Annual Financial Statements from 5 companies. 


## Technologies Used

- **Programming Language:** Python
- **Web Framework:** Streamlit
- **Machine Learning Libraries:** Scikit-learn, XGBoost, RandomForest, LGBM
- **Data Analysis and Visualization:** Pandas, NumPy, Matplotlib, Seaborn
- **LLM Model:** TinyLlama-1.1B
- **Document Retrieval:** ChromaDB



## Model Development Process

### User Classification
User Classification model was developed through the following steps:

1. **Importing the Dependencies**
2. **Exploratory Data Analysis (EDA)**
3. **Data Preprocessing**
    - Handling missing values
    - Handling outliers
    - Label encoding/One-hot encoding
    - Standardizing the data
4. **Model Selection**
    - Selected the most common 3 classification models
    - Trained each model and checked cross-validation scores (ROC-AUC)
    - Choose the best models based on cross-validation scores
5. **Model Building and Evaluation**
    - Selected best features using Backward Elimination
    - Performed hyperparameter tuning using Grid Search CV
    - Built the final model with the best hyperparameters and features
    - Evaluated the model using ROC-AUC Scores
    - Analyze feature Importance

### Financial Chatbot

#### RAG Architecture
The following diagram explains the information flow when a user asks a question:

<img src="Example Q&A/graph.png" alt="Alt Text" width="1000">

**Step-by-Step Process:**

1. Document Ingestion: PDFs are loaded and split into chunks.

2. Embedding & Storage: Text chunks are embedded using HuggingFaceEmbeddings and stored in ChromaDB.

3. Query Processing: When a user enters a query, the system retrieves the most relevant text chunks from ChromaDB.

4. LLM Response Generation: The retrieved context and user query are fed into the LLM to generate an answer.

5. Response Display: The chatbot returns an AI-generated answer along with cited sources.

#### Assumptions & Trade-offs

**Assumptions:**
- Financial statements are in text-based PDF format (scanned PDFs require OCR processing).
- The user queries are in English and related to financial documents.
- The chatbot can return partial answers if exact data is unavailable.

**Trade-offs:**

- Using a lighter LLM (e.g., Gemma-2B, TinyLlama) means faster inference but may sacrifice some reasoning capabilities compared to larger models (e.g., LLaMA-2 13B).
- ChromaDB retrieval focuses on semantic similarity, which may sometimes retrieve less relevant passages for edge cases.

## Summaries

### User Classification

1. Data Understanding is an **important part of getting insight** from data so that we can carry out **better data processing** at the next stage
2. There is **no perfect method** of handling outliers or handling missing values so we have to experiment it also with some methods
3. **XGBoost** is the best model, but the performance is still not very high (ROC-AUC 0.6816). While LGBM is a **close second** and might be worth further tuning.
4. **Further improvements** can be made through **feature engineering, handling class imbalance, and advanced hyperparameter tuning**.

### Financial Chatbot

1. **Challenges Faced:**
- Handling multi-PDF queries effectively requires better metadata tracking.
- ptimizing retrieval to avoid irrelevant chunks being fetched.
- Balancing LLM size vs. response quality (lightweight models vs. accuracy trade-off).

2. **Future Enhancement:**
✅ OCR support for scanned PDFs.
✅ Multilingual support for financial document analysis.
✅ Better filtering techniques (e.g., keyword + embedding hybrid search).
✅ Fine-tuning a model specifically on financial statements for better accuracy.

## Example Input Outputs
<img src="Example Q&A/example_user.png" alt="Alt Text" width="1000">
<img src="Example Q&A/example_chatbot.png" alt="Alt Text" width="1000">


## Contributing

Feel free to contribute by improving retrieval quality, fine-tuning a better model, or adding more user-friendly features in Streamlit.

## Contact

If you have any questions or suggestions, feel free to contact me at ekasurya1410@gmail.com.
