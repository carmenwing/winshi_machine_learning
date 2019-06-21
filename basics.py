import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("raw_sortate.csv") 

#Afiseaza primele 10 randuri din fisier
print("Afiseaza primele 10 randuri din fisier")
print(df[:10]) 

#Calculeaza diferite tipuri de corelatii
print("Pearson Correlation")
print(df.corr(method ='pearson'))
print("Kendall Correlation")
print(df.corr(method ='kendall'))
print("Spearman Correlation")
print(df.corr(method ='spearman'))

#Calculeaza percentile
print("Mean, max, percentile")
perc =[.20, .40, .60, .50, .75, .80, .95] 
include =['object', 'float', 'int'] 
desc = df[['PM1','PM2.5','PM10']].describe(percentiles = perc, include = include) 
print(desc) 

#VERIFICA MEDIAN CU PERCENTILA 50
print("VERIFICA MEDIAN CU PERCENTILA 50 pentru PM2.5")
print(df['PM2.5'].median())

#Deviatie standard a unui parametru
print("Deviatie Standard PM2.5")
print(df['PM2.5'].std())

#Varianta unui parametru
print("Varianta PM2.5")
print(df['PM2.5'].var())

#grafic
plt.plot(df["HUM"], df["PM2.5"], 'ro--')
plt.show()
