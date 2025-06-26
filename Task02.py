import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('train.csv')
df['Age'].fillna(df['Age'].median(), inplace=True)
df.drop(columns=['Cabin'], inplace=True)
df.dropna(subset=['Embarked'], inplace=True)

df['Sex'] = df['Sex'].map({'male':0, 'female':1})
df['Embarked'] = df['Embarked'].map({'S':0, 'C':1, 'Q':2})
df['Prefix'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
prefix_map = {"Mr":0,"Miss":1,"Mrs":2,"Master":0}
df['Prefix'] = df['Prefix'].map(prefix_map).fillna(3)

sns.barplot(x='Sex', y='Survived', data=df); plt.show()
sns.barplot(x='Pclass', y='Survived', data=df); plt.show()
sns.histplot(df['Age'], bins=30, kde=True); plt.show()
sns.boxplot(x='Survived', y='Fare', data=df); plt.show()
sns.heatmap(df.select_dtypes(include=['int64','float64']).corr(), annot=True, cmap='coolwarm'); plt.show()
