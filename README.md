# 🎬 Machine Learning Internship - Movie Recommendation System

**Project Name:** `Cognetix_MovieRecommender`  
**Internship Domain:** Machine Learning / Recommendation Systems  
**Organization:** @Cognetix Technology  

---

## 🎯 Objective

The objective of this final capstone project was to develop a **Content-Based Movie Recommendation System** capable of suggesting personalized movies based on genre similarity. This system was built to handle a large dataset efficiently and demonstrate real-time deployment.

---

## ⚙️ Functional Requirements & Key Steps

This project fully satisfied all core requirements, including the optional Streamlit deployment:

1. **Data Processing:** Loaded, cleaned, and merged `ratings.csv` and `movies.csv` from the MovieLens dataset.  
2. **Content-Based Filtering:** Used **TF-IDF Vectorization** on movie genres to create feature vectors, effectively representing each movie's genre profile.  
3. **Similarity Calculation:** Computed the **Cosine Similarity Matrix** to measure the genre-based similarity between all movie pairs.  
4. **Efficiency Handling:** Implemented **data sampling** and **Streamlit caching** to manage the large size of the similarity matrix and prevent `MemoryError` during deployment.  
5. **Recommendation Engine:** Developed a function to retrieve the top 10 most similar movies based on a user's selected film.  
6. **Visualization:** Plotted the **Top 10 Most Popular Genres** (by movie count) using a bar chart.  
7. **Deployment (Optional):** Implemented an interactive web interface using **Streamlit** to demonstrate real-time recommendations.  

---

## 📊 Results and Analysis

The **Content-Based Filtering** approach proved highly effective. The model successfully returned films sharing identical genres with a **perfect Similarity Score (1.0)**, confirming the accuracy of the genre matching.

### Evaluation Summary

Since this model recommends items (not predicting numerical ratings), evaluation focused on relevance:

- **Metric:** Subjective Relevance and Conceptual Precision@K  
- **Result:** The system consistently provides high-quality, genre-aligned recommendations, demonstrating a successful implementation of the core content-based logic.  

---

## 🚀 Model Deployment (Streamlit)

The recommendation engine is deployed via a simple, interactive web application, allowing users to select any movie and instantly receive a list of 10 similar titles.

**Execution Command:**
```bash
streamlit run app.py
```
---

## 🛠️ Technology Stack
- **Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn (TF-IDF, Cosine Similarity), Matplotlib, Seaborn, Streamlit  
- **Dataset Source:** MovieLens 20M Dataset (using sampled `movies.csv` and `ratings.csv`)  

---

## 👩‍💻 Project Done By
**Hemavarni S**
