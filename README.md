# keyword_clustering_easy_demo
This repository contains an advanced Python script designed for keyword clustering using state-of-the-art Natural Language Processing (NLP) techniques.
The script is designed for professionals in SEO, data science, and machine learning who need efficient and scalable solutions for text clustering and analysis.

The script includes functionalities such as SentenceTransformer for generating semantic embeddings, BERTopic for advanced topic modeling and clustering, and DataFrame integration for processing and appending cluster labels. Optional preprocessing techniques like lemmatization and stemming are also supported for improving clustering accuracy. Additionally, the script is optimized for handling large datasets by dividing data into blocks, ensuring efficient memory usage.

Key features of the script include embedding generation with support for various models, dynamic topic labeling for enhanced interpretability, and seamless output to Excel files. The script is particularly effective for Italian and multilingual datasets, making it versatile for various professional use cases.

Requirements for running the script include Python 3.7+, along with libraries like pandas, numpy, sentence-transformers, bertopic, and openpyxl. Optional tools like spaCy or NLTK can also be utilized for preprocessing tasks.

To use this script, users need to prepare a pandas DataFrame with a column containing the keywords or text to be clustered. After configuring the desired SentenceTransformer model and BERTopic parameters, the script can be executed to retrieve clustering results in an Excel file.

This repository offers a powerful solution for tasks such as keyword clustering for SEO analysis, semantic topic modeling for research, intent analysis in multilingual datasets, and preprocessing for machine learning models. By leveraging the latest advancements in NLP, the script provides an efficient and scalable approach to text clustering and topic analysis.

## Features
Generate semantic embeddings using SentenceTransformer with support for multilingual and Italian-specific models.
Perform topic modeling and clustering with BERTopic, including dynamic topic labeling for better interpretability.
Integrate seamlessly with pandas DataFrames for processing and appending cluster labels.
Optimize memory usage and handle large datasets by processing in blocks.
Export clustering results directly to Excel files.

## Requirements
Python 3.7 or higher

Libraries:
pandas
numpy
sentence-transformers
bertopic
openpyxl
Optional: spaCy or NLTK for preprocessing

## How to Use
Install the required libraries using pip install -r requirements.txt.
Prepare a pandas DataFrame with a column named Cleaned containing the text or keywords to cluster.
Configure the SentenceTransformer model and BERTopic parameters in the script.
Execute the script to generate clustering results. Results will be saved as an Excel file.

## Applications
Keyword clustering for SEO analysis
Semantic topic modeling for research
Intent analysis in multilingual datasets
Preprocessing for machine learning models

## Acknowledgments
This script leverages cutting-edge NLP libraries like sentence-transformers and BERTopic to provide a robust and scalable solution for clustering text data. It is optimized for professional use cases, especially in Italian and multilingual contexts.
