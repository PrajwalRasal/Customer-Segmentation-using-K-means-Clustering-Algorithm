# Customer-Segmentation-using-K-means-Clustering-Algorithm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

# Load dataset
df = pd.read_csv("Mall_Customers.csv")

# Rename columns
df.rename(columns={'Genre': 'Gender'}, inplace=True)

# Drop unnecessary column
df.drop(['CustomerID'], axis=1, inplace=True)

# Display basic information
print(df.head())
print(df.shape)
print(df.describe())
print(df.dtypes)
print(df.isnull().sum())

# Visualize data distribution
plt.figure(1, figsize=(15, 6))
n = 0
for x in ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']:
    n += 1
    plt.subplot(1, 3, n)
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    sns.histplot(df[x], bins=20, kde=True)
    plt.title('Histogram of {}'.format(x))
plt.show()

# Gender distribution
plt.figure(figsize=(15, 5))
sns.countplot(y='Gender', data=df)
plt.show()

# Violin plots
plt.figure(1, figsize=(15, 7))
n = 0
for cols in ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']:
    n += 1
    plt.subplot(1, 3, n)
    sns.set(style="whitegrid")
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    sns.violinplot(x=cols, y='Gender', data=df)
    plt.ylabel('Gender' if n == 1 else '')
    plt.title('Violin Plot')
plt.show()

# Age distribution
age_groups = {
    "18-25": (18, 25),
    "26-35": (26, 35),
    "36-45": (36, 45),
    "46-55": (46, 55),
    "55+": (56, np.inf)
}
agey = [len(df.Age[(df.Age >= age_min) & (df.Age <= age_max)]) for age_min, age_max in age_groups.values()]
plt.figure(figsize=(15, 6))
sns.barplot(x=list(age_groups.keys()), y=agey, palette="mako")
plt.title("Number of Customers by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Number of Customers")
plt.show()

# Spending Score distribution
ss_groups = {
    "1-20": (1, 20),
    "21-40": (21, 40),
    "41-60": (41, 60),
    "61-80": (61, 80),
    "81-100": (81, 100)
}
ssy = [len(df["Spending Score (1-100)"][(df["Spending Score (1-100)"] >= ss_min) & (df["Spending Score (1-100)"] <= ss_max)]) for ss_min, ss_max in ss_groups.values()]
plt.figure(figsize=(15, 6))
sns.barplot(x=list(ss_groups.keys()), y=ssy, palette="rocket")
plt.title("Spending Scores Distribution")
plt.xlabel("Score Range")
plt.ylabel("Number of Customers")
plt.show()

# Annual Income distribution
ai_groups = {
    "$ 0 - 30,000": (0, 30),
    "$ 30,001 - 60,000": (31, 60),
    "$ 60,001 - 90,000": (61, 90),
    "$ 90,001 - 120,000": (91, 120),
    "$ 120,000 - 150,000": (121, 150)
}
aiy = [len(df["Annual Income (k$)"][(df["Annual Income (k$)"] >= ai_min) & (df["Annual Income (k$)"] <= ai_max)]) for ai_min, ai_max in ai_groups.values()]
plt.figure(figsize=(15, 6))
sns.barplot(x=list(ai_groups.keys()), y=aiy, palette="Spectral")
plt.title("Annual Income Distribution")
plt.xlabel("Income Range")
plt.ylabel("Number of Customers")
plt.show()

# K-Means Clustering
X1 = df.loc[:, ["Age", "Spending Score (1-100)"]].values

wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init="k-means++")
    kmeans.fit(X1)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(12, 6))
plt.grid()
plt.plot(range(1, 11), wcss, linewidth=2, color="red", marker="8")
plt.xlabel("K Value")
plt.ylabel("WCSS")
plt.show()

kmeans = KMeans(n_clusters=4)
label = kmeans.fit_predict(X1)
print(label)
print(kmeans.cluster_centers_)

plt.scatter(X1[:, 0], X1[:, 1], c=kmeans.labels_, cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='black')
plt.title("Clusters of Customers (Age vs Spending Score)")
plt.xlabel('Age')
plt.ylabel('Spending Score (1-100)')
plt.show()

X2 = df.loc[:, ["Annual Income (k$)", "Spending Score (1-100)"]].values

wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init="k-means++")
    kmeans.fit(X2)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(12, 6))
plt.grid()
plt.plot(range(1, 11), wcss, linewidth=2, color="red", marker="8")
plt.xlabel("K Value")
plt.ylabel("WCSS")
plt.show()

kmeans = KMeans(n_clusters=5)
label1 = kmeans.fit_predict(X2)
print(label1)
print(kmeans.cluster_centers_)

plt.scatter(X2[:, 0], X2[:, 1], c=kmeans.labels_, cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='black')
plt.title("Clusters of Customers (Annual Income vs Spending Score)")
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()

X3 = df.iloc[:, 1:]

wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init="k-means++")
    kmeans.fit(X3)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(12, 6))
plt.grid()
plt.plot(range(1, 11), wcss, linewidth=2, color="red", marker="8")
plt.xlabel("K Value")
plt.ylabel("WCSS")
plt.show()

kmeans = KMeans(n_clusters=5)
Clusters = kmeans.fit_predict(X3)
print(Clusters)
print(kmeans.cluster_centers_)

df["label"] = Clusters

# 3D plot
fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(111, projection='3d')
colors = ['blue', 'red', 'green', 'orange', 'purple']
for i in range(5):
    ax.scatter(df.Age[df.label == i], df["Annual Income (k$)"][df.label == i], df["Spending Score (1-100)"][df.label == i], c=colors[i], s=60, label=f'Cluster {i}')
ax.view_init(30, 185)
plt.xlabel("Age")
plt.ylabel("Annual Income (k$)")
ax.set_zlabel("Spending Score (1-100)")
plt.legend()
plt.show()
