# Machine-Learning
This repo consists of Algorithms of Data Analysis using scikit learn which is based on Supervised learning and unsupervised learning.

Supervised learning and unsupervised learning are two fundamental categories of machine learning techniques, each with its own purpose and characteristics.

Supervised Learning:
Supervised learning is a type of machine learning where the algorithm learns from labeled training data. In supervised learning, the dataset used for training includes both input features and corresponding target labels (desired outputs). The goal of supervised learning is to learn a mapping from input features to output labels so that the model can make accurate predictions on new data.

Common types of supervised learning tasks include:

Classification: The goal is to predict a categorical label or class for each input instance. Examples include email spam detection, image classification, and sentiment analysis.
Regression: The goal is to predict a continuous numerical value for each input instance. Examples include predicting house prices, temperature forecasts, and stock prices.

In supervised learning, the model is trained using the provided labeled data, and its performance is evaluated based on its ability to accurately predict the labels on new data.

Unsupervised Learning:
Unsupervised learning involves learning patterns and structures from unlabeled data. Unlike supervised learning, there are no explicit target labels provided in the training data. Instead, the algorithm attempts to find inherent patterns, relationships, or groupings within the data without any prior knowledge of the expected outputs.

Common types of unsupervised learning tasks include:

Clustering: The goal is to group similar instances together based on some similarity metric. Examples include customer segmentation and image segmentation.
Dimensionality Reduction: The goal is to reduce the number of features in the data while preserving its important information. Principal Component Analysis (PCA) is a popular technique in this category.
Anomaly Detection: The goal is to identify rare instances that deviate significantly from the normal one. This is useful for fraud detection and identifying defects in manufacturing processes.

In unsupervised learning, the model tries to discover the underlying structure of the data without explicit guidance, and the evaluation can be more challenging since there are no predefined target labels to compare against.

# Supervised learning

There are several machine learning algorithms in the category of supervised learning. Each algorithm has its own characteristics and is suitable for different types of problems. Here are some commonly used supervised learning algorithms:

Linear Regression: A regression algorithm used for predicting a continuous numeric value. It fits a linear relationship between input features and the target variable.

Logistic Regression: Despite its name, logistic regression is used for binary classification problems. It estimates the probability that a given input belongs to a certain class.

Decision Trees: Decision trees split the data into subsets based on the value of input features. They're used for both classification and regression tasks.

Random Forest: An ensemble method that builds multiple decision trees and combines their predictions to improve accuracy and reduce overfitting.

Support Vector Machines (SVM): SVM finds a hyperplane that best separates different classes in the data. It can be used for both binary and multiclass classification.

K-Nearest Neighbors (KNN): KNN assigns a class label to a data point based on the class labels of its k nearest neighbors in the training data.

Naive Bayes: Based on Bayes' theorem, Naive Bayes is used for classification tasks. It assumes that features are independent given the class label.

Neural Networks: Deep learning neural networks, including feedforward, convolutional, and recurrent networks, can be used for various supervised learning tasks, including image and text classification.

Gradient Boosting Algorithms: Algorithms like XGBoost, LightGBM, and CatBoost create an ensemble of weak models that are iteratively improved to produce a strong predictive model.

These are just a few examples of supervised learning algorithms. The choice of algorithm depends on factors like the nature of the problem, the amount of available data, the type of features, and the desired level of interpretability and performance. 

# Unsupervised learning

There are several machine learning algorithms that belong to the category of unsupervised learning. These algorithms are used to discover patterns, relationships, and structures within unlabeled data. Here are some commonly used unsupervised learning algorithms:

K-Means Clustering: K-Means is a popular clustering algorithm that groups similar data points into clusters. It aims to minimize the distance between data points within a cluster and maximize the distance between clusters.

Hierarchical Clustering: This algorithm creates a hierarchy of clusters by successively merging or splitting existing clusters based on their similarity. It results in a dendrogram that can help in understanding the data's hierarchy.

DBSCAN (Density-Based Spatial Clustering of Applications with Noise): DBSCAN groups together data points that are closely packed and identifies noise points as outliers.

Principal Component Analysis (PCA): PCA is a dimensionality reduction technique that transforms data into a lower-dimensional space while preserving as much of the original variance as possible.

Independent Component Analysis (ICA): ICA is used for separating a multivariate signal into additive, independent components.

Autoencoders: These are neural network architectures used for unsupervised learning. They aim to reconstruct the input data, typically by compressing it into a lower-dimensional representation and then decoding it back to the original input.

Non-negative Matrix Factorization (NMF): NMF factorizes a matrix into two lower-dimensional matrices with the constraint that all elements are non-negative. It's often used for feature extraction and topic modeling.

Gaussian Mixture Models (GMM): GMM represents a mixture of multiple Gaussian distributions, often used for modeling complex data distributions.

Singular Value Decomposition (SVD): SVD decomposes a matrix into three matrices, which can be used for dimensionality reduction and noise reduction.

Word Embeddings: Techniques like Word2Vec and GloVe create dense vector representations of words based on their co-occurrence patterns in text data.

These algorithms play a crucial role in identifying patterns, clusters, and latent structures within data. The choice of algorithm depends on the nature of the data and the goals of the analysis. Unsupervised learning is particularly useful for exploratory data analysis, data preprocessing, and understanding the underlying structure of the data.
