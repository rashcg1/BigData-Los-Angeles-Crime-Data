# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 19:27:42 2018

@author: Rashu
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
#matplotlib inline

#arrest = pd.read_csv(r'D:\los-angeles-crime-arrest-data\arrest-data-from-2010-to-present.csv')
crime= pd.read_csv(r'D:\los-angeles-crime-arrest-data\crime-data-from-2010-to-present.csv')



crime.shape
crime.describe()
crime.isna().sum()
c1=crime.drop(columns=['DR Number','Area ID','Reporting District','MO Codes',
                       'Premise Code','Weapon Used Code','Weapon Description',
                       'Premise Description','Crime Code 2','Crime Code 3',
                       'Crime Code 4','Cross Street'])

c1.columns
#Null values in the data set
c1.isna().sum()
#only 8 % of the total records in age column is null which is trival. Hence deleting the records.
c1=c1.dropna(subset=['Victim Age'])



#to identify the age group involved in the crime, first dicretize the age column to 10 different bins.




plt.hist(c1.loc['Victim Age'].values, bins=bin, edgecolor="k")
plt.xticks(bin)

plt.show()
c2 = pd.concat([c1['Victim Age'],category],axis = 1)
c2.isna().any()
c1.columns
c2.head()
c3 = pd.concat([c2,c1['Crime Category']],axis = 1)
c3.head()
c3.dtypes

sns.set(style="darkgrid")
g = sns.FacetGrid(c3, col='Crime Category',margin_titles=True,col_wrap=3)
g.map(plt.hist, 'Victim Age')
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Age vs Crime Category') # can also get the figure from plt.gcf()
g.add_legend()

labelEncoder = LabelEncoder()
labelEncoder.fit(c3['Crime Category'])
c3['Crime Category'] = labelEncoder.transform(c3['Crime Category'])

labelEncoder.fit(c3['Age Group'])
c3['Age Group'] = labelEncoder.transform(c3['Age Group'])

#Applying k-means on the c3 dataset.
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5, random_state=0).fit(c3.iloc[:,[0,2]])
y_means=kmeans.predict(c3.iloc[:,[0,2]])

fig = plt.figure(figsize=(12, 8))
plt.scatter(c3.iloc[:, 0], c3.iloc[:, 2], c=y_means,s=100)

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=100, alpha=0.5);



labels = kmeans.predict(c3.iloc[:,[0,2]])
centroids = kmeans.cluster_centers_

fig = plt.figure(figsize=(12, 8))

colors = list(map(lambda x: colmap[x+1], labels))

plt.scatter(c3['Victim Age'], c3['Crime Category'], cmap='viridis', c=labels)
for idx, centroid in enumerate(centroids):
    plt.scatter(*centroid, color=colmap[idx+1])
plt.xlim(0, 80)
plt.ylim(0, 80)
plt.show()















#there are 139 unique crime codes in the data set
c4=c1['Crime Code'].unique()
for counter,value in enumerate(c1['Crime Code'],1):
    print(counter,value)

len(c1['Crime Code Description'].unique())
c1.groupby('Crime Code Description').count()


bin1=['Theft',]
category = pd.cut(c1['Crime Code Description'],bin)


#Then roll up the similar crime codes to a series to bring down the number of clusters.

#Grouping Crime in categories:
#Crime code 121 and 122 are rape,236,626 860 762 821 820 956-815 810 806 805 840 830sexual abuse
#Crime Code 110 criminal homocide

#210 220 451  452-Robbery
#230 231,624-623 625 627Assult
#235,237 812 813- 922 880 954 865 870child abuse
#251,250  756 753 435 436 931 952 shooting
#310 320 330 410 888 850 933 622burglary
#331 341 343 345,347,349,350,351,352,354 420,421,442,668,666,664 520 510 649 450 670 480 441  443 444 445 446
# 662 352 474 473 471 470 485 487 475theft


#740 648-924 vandalism
#Fraud:950 951 653 661 660 654
#946 932 763 886 949-#434 false imprisonment #439 false police report
#944 conspiracy
#882,890 884
#948 bigamy(second marriage) miscellaneous ignore
#920 910kiddnapping
#901 900 902 903 651 942 652VIOLATION OF RESTRAINING ORDER-#437 Resisiting arrest 

#943-cruelity to animals

#453:drunk ignore1entry

#438  353reckless driving


#113-only 1 entry-ignoring it,353

bin1=["Theft" ,"Assult","Burglary","Shooting","Vandalism","SexualAbuse","ChildAbuse","Robbery","Fraud","GovtVoilation","Miscellaneous"]





