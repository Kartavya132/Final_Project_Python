from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

print("=" * 50)
print("Loading Titanic Dataset...")
print("=" * 50)

train = pd.read_csv("database/train.csv")
test = pd.read_csv("database/test.csv")
gender_submission = pd.read_csv("database/gender_submission.csv")

print("\n" + "=" * 50)
print("TITANIC DATASET OBSERVATIONS")
print("=" * 50)

print("\nTrain Dataset Shape:", train.shape)
print("Test Dataset Shape:", test.shape)

print("\n========== TRAIN DATA INFO ==========")
print(train.info())

print("\n========== FIRST 5 ROWS ==========")
print(train.head())

print("\n========== MISSING VALUES ==========")
print(train.isnull().sum())

print("\n========== STATISTICAL SUMMARY ==========")
print(train.describe(include="all"))

print("\n========== SURVIVAL RATE ==========")
survival_rate = train["Survived"].mean() * 100
print(f"Overall Survival Rate: {survival_rate:.2f}%")

sns.set_theme(style="whitegrid")
plt.rcParams.update(
    {
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    }
)

train = train.copy()
train["Age"] = train["Age"].fillna(train["Age"].median())
train["Embarked"] = train["Embarked"].fillna(train["Embarked"].mode()[0])
train["FamilySize"] = train["SibSp"] + train["Parch"] + 1


fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
sns.countplot(
    data=train,
    x="Survived",
    palette=["red", "green"],
    saturation=0.95,
    ax=ax,
)
ax.set_title("Survival Count")
ax.set_xlabel("Survival")
ax.set_xticks([0, 1])
ax.set_xticklabels(["Did Not Survive", "Survived"])
ax.set_ylabel("Number of Passengers")
ax.spines[["top", "right"]].set_visible(False)
for container in ax.containers:
    ax.bar_label(container, fmt="%d", padding=3)
plt.show()

fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
sns.barplot(
    data=train,
    x="Sex",
    y="Survived",
    estimator="mean",
    errorbar=None,
    palette=["blue", "orange"],
    ax=ax,
)
ax.set_title("Survival Rate by Gender")
ax.set_xlabel("Gender")
ax.set_ylabel("Survival Rate")
ax.spines[["top", "right"]].set_visible(False)
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", padding=3)
plt.show()

fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
sns.barplot(
    data=train,
    x="Pclass",
    y="Survived",
    estimator="mean",
    errorbar=None,
    palette="viridis",
    ax=ax,
)
ax.set_title("Survival Rate by Passenger Class")
ax.set_xlabel("Passenger Class")
ax.set_ylabel("Survival Rate")
ax.spines[["top", "right"]].set_visible(False)
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", padding=3)
plt.show()

fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
sns.histplot(
    data=train,
    x="Age",
    hue="Survived",
    bins=25,
    kde=True,
    palette=["red", "green"],
    alpha=0.8,
    ax=ax,
)
ax.set_title("Age Distribution by Survival")
ax.set_xlabel("Age")
ax.set_ylabel("Count")
ax.spines[["top", "right"]].set_visible(False)
plt.show()

fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
sns.histplot(train["Age"], bins=25, color="#4c78a8", edgecolor="white", ax=ax)
ax.set_title("Age Distribution")
ax.set_xlabel("Age")
ax.set_ylabel("Count")
ax.spines[["top", "right"]].set_visible(False)
plt.show()

fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
sns.histplot(train["Fare"], bins=30, color="#f28e2b", edgecolor="white", ax=ax)
ax.set_title("Fare Distribution")
ax.set_xlabel("Fare")
ax.set_ylabel("Count")
ax.spines[["top", "right"]].set_visible(False)
plt.show()

fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
sns.barplot(
    data=train,
    x="Embarked",
    y="Survived",
    estimator="mean",
    errorbar=None,
    order=["S", "C", "Q"],
    palette="mako",
    ax=ax,
)
ax.set_title("Survival Rate by Embarkation Port")
ax.set_xlabel("Embarked Port")
ax.set_ylabel("Survival Rate")
ax.spines[["top", "right"]].set_visible(False)
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", padding=3)
plt.show()

fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
sns.barplot(
    data=train,
    x="FamilySize",
    y="Survived",
    estimator="mean",
    errorbar=None,
    palette="rocket",
    ax=ax,
)
ax.set_title("Survival Rate by Family Size")
ax.set_xlabel("Family Size")
ax.set_ylabel("Survival Rate")
ax.spines[["top", "right"]].set_visible(False)
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", padding=3)
plt.show()

numeric_columns = train.select_dtypes(include=["number"]).drop(
    columns=["PassengerId"], errors="ignore"
)

fig, ax = plt.subplots(figsize=(10, 7), dpi=120)
sns.heatmap(
    numeric_columns.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5,
    square=False,
    ax=ax,
)
ax.set_title("Correlation Heatmap")
plt.show()

print("Analysis Completed Successfully!")
