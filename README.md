# ProductPriceAI
Machine learning and LLM-based system for predicting product prices from textual descriptions.

**ProductPriceAI | End-to-End Machine Learning & NLP-Based Price Prediction**  
*Python, PyTorch, Scikit-learn, XGBoost, NLP, Groq API, Hugging Face, Git, Google Colab*

- Developed an end-to-end machine learning pipeline to predict e-commerce product prices from unstructured product descriptions, leveraging the Amazon Reviews 2023 dataset containing over **8.2 million product records** across eight categories.
- Designed scalable data curation and preprocessing workflows, including data quality assessment, price filtering, text cleaning, feature engineering, and dataset partitioning to prepare approximately **800,000 training samples, 10,000 validation set and 10,000 test set**.
- Built an automated NLP summarization pipeline using **Groq's Batch API and GPT-OSS-20B** to transform lengthy product descriptions into concise, standardized representations for downstream modeling.
- Developed and evaluated multiple price prediction models, including **linear regression, Random Forest, XGBoost, and a PyTorch-based neural network with sparse text representations**, incorporating CountVectorizer and HashingVectorizer for text feature extraction.
- Implemented a reusable model evaluation framework to compare predictive performance using **mean absolute error (MAE), mean squared error (MSE), RMSE, and confidence intervals**, with visualizations for prediction accuracy and error distributions.
- Published curated datasets to **Hugging Face** and maintained reproducible model development, experimentation, and version control workflows using **Google Colab and GitHub**.
