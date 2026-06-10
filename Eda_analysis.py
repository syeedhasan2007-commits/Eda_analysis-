import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("StudentsPerformance.csv")

print(df.head())

print(df.info())

print(df.describe())


print(df.isnull().sum())

plt.figure(figsize=(8,5))

sns.histplot(df["math score"], bins=20)

plt.title("Distribution of Math Scores")

plt.savefig("hist_math_score.png")

plt.show()

scores = df[[
    "math score",
    "reading score",
    "writing score"
]]

corr = scores.corr()

print(corr)

plt.figure(figsize=(6,5))

sns.heatmap(
    corr,
    annot=True,
    cmap="Blues"
)

plt.title("Correlation Heatmap")

plt.savefig("correlation_heatmap.png")

plt.show()

plt.figure(figsize=(8,5))

sns.boxplot(
    data=scores
)

plt.title("Score Distribution")

plt.savefig("boxplot_scores.png")

plt.show()

print("Average Math Score:",
      df["math score"].mean())

print("Average Reading Score:",
      df["reading score"].mean())

print("Average Writing Score:",
      df["writing score"].mean())