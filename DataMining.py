# -*- coding: utf-8 -*-
"""
Created on Sun May 21 04:56:23 2023

@author: EL Rowad
"""

from tkinter import *
from tkinter import messagebox
import tkinter.scrolledtext as st
import tkinter.font as font
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import seaborn as snsimport
import seaborn as sns  
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt 
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.cluster import contingency_matrix
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure


figure = Figure(figsize=(5,5), dpi=80)
ax = figure.add_subplot(111)

def km():
    
    if(int(testValue.get()) >= 3 and int(testValue.get()) <= 9):
        df=pd.read_csv("Mall_Customers.csv")
        plt.scatter(df.Age,df['Spending Score (1-100)'])
        plt.show()
        
        k=int(testValue.get())
        km=KMeans(n_clusters=k)
        y_predicted=km.fit_predict(df[['Age','Spending Score (1-100)']])
        labels = km.labels_
        df['cluster']=y_predicted
        centroids= km.cluster_centers_
        ax.clear()
        colors = ['#612900', '#030303', '#118f7a', '#f77f00', '#6a191b','#a74d0f','#0c4524','#d4ac90']
        for i in range(k):
            df0=df[df.cluster==i]
            ax.scatter(df0.Age,df0['Spending Score (1-100)'],c=colors[i],label='Group {}'.format(i+1))
            ax.scatter(centroids[i, 0], centroids[i, 1], s=200, marker='*', color='red')
        ax.legend()
        ax.set_title('K-Means Clustering Results')
        ax.set_xlabel('Age')
        ax.set_ylabel('Spending Score (1-100)')
        canvas.draw()
    else:
        messagebox.showerror("invalid range", "Choose the optimal k from range [3:9]")
        return
    
    def elb():
        ax.clear()
        sse=[]
        k_rng=range(1,10)
        for k in k_rng:
            km=KMeans(n_clusters=k)
            km.fit(df[['Age','Spending Score (1-100)']])
            sse.append(km.inertia_)
            
        ax.plot(k_rng,sse)
        ax.set_title('Elbow Method')
        ax.set_xlabel('K')
        ax.set_ylabel('Sum of squared error')
        canvas.draw() 
        
    
    
    
    def eva():
       
       sil = silhouette_score(df[['Age','Spending Score (1-100)']],y_predicted) #1 - (cohesion / seperation)
       dav = davies_bouldin_score(df[['Age','Spending Score (1-100)']],y_predicted) #cohesion
       cal = calinski_harabasz_score(df[['Age','Spending Score (1-100)']],y_predicted) #seperation
       ss = km.inertia_
       
       
       X=df[['Age','Spending Score (1-100)']]
       similarity_matrix = cosine_similarity(X)
       indices = np.argsort(km.labels_)
       # Sort the samples by their cluster labelsindices = np.argsort(km.labels_)
       sorted_similarity_matrix = similarity_matrix[indices][:, indices]
       
       # Create a heatmap of the sorted similarity matrix
       fig, ax = plt.subplots()
       im = ax.imshow(sorted_similarity_matrix, cmap='viridis')
       ax.set_xticks(np.arange(len(X)))
       ax.set_yticks(np.arange(len(X)))
       ax.set_xticklabels(indices)
       ax.set_yticklabels(indices)
       plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
       plt.colorbar(im)
       ax.set_title("Similarity matrix of k-means clusters")
       plt.show() 
       

       
       X=df[['Age','Spending Score (1-100)','cluster']]
       incidence_matrix = np.zeros((X.shape[0], X.shape[0]), dtype=int)
       k=0;
       for i in X['cluster']:
           o=0    
           for j in X['cluster']:
               if i==j:            
                   incidence_matrix[k][o]=1
               o=o+1        
           k=k+1 
           
       print(incidence_matrix)
       X_values=np.array(similarity_matrix)
       Y_values=np.array(incidence_matrix)
       # Calculate the correlation coefficient between X and Y
       corr_matrix = np.corrcoef(X_values, Y_values)
       corr_coef = corr_matrix[0, 1]
       # Print the correlation coefficient
       corr = corr_coef

       print("Correlation coefficient:", corr_coef)

       ss_label.configure(text='SSE Score {}'.format(round(ss,3)),font=11)
       sil_label.configure(text='Silhouette Score {}'.format(round(sil,3)),font=11)
       dav_label.configure(text='Davies Bouldin Score {}'.format(round(dav,3)),font=11)
       cal_label.configure(text='Calinski Harabasz Score {}'.format(round(cal,3)),font=11)
       corr_label.configure(text='Correlation coefficient {}'.format(round(corr,3)),font=11)

       
    elb_butt = Button(root,text="Show Elbow", command=elb , bg='#473337',fg="white",
                     padx=30,pady=10, font=("Fira Sans", 12))
    elb_butt.place(x=520,y=550)
       
    eval_butt = Button(root,text="Evaluation", command=eva , bg='#473337',fg="white",
                     padx=30,pady=10, font=("Fira Sans", 12))
    eval_butt.place(x=530,y=620)

    
   
root = Tk()
root.title('Data Mining Project')
root.configure(bg='#230f19')
root.geometry("1200x1200")
f = font.Font(weight="bold")   

lbl1= Label(root , text="Hello on K-mean algorithm" , font=("Fira Sans", 15),
            bg="#230f19", fg="white",pady=20)
lbl1.pack() 
lbl1= Label(root , text="Input the k clusters" , font=("Fira Sans", 15),
            bg="#230f19", fg="white")
lbl1.place(x=100,y=50)
testValue=Entry(root)
testValue.place(x=300,y=55)

km_butt = Button(root,text="Clust Data", command=km , bg='#473337',fg="white",
                 padx=30,pady=10, font=("Fira Sans", 12))
km_butt.pack()


canvas = FigureCanvasTkAgg(figure, root)
canvas.draw()
canvas.get_tk_widget().pack(pady=10)
toolbar = NavigationToolbar2Tk(canvas, root)
toolbar.update()
canvas.get_tk_widget().pack() 


ss_label = Label(root, text='SSE Score:', bg='#473337',fg="white", font=11)
ss_label.place(x=50, y=150)

sil_label = Label(root, text='Silhouette Score:', bg='#473337',fg="white",font=11)
sil_label.place(x=50, y=250)

dav_label = Label(root, text='Davies Bouldin Score:', bg='#473337',fg="white",font=11)
dav_label.place(x=50, y=350)

cal_label = Label(root, text='Calinski Harabasz Score:', bg='#473337',fg="white",font=11)
cal_label.place(x=50, y=450)

corr_label = Label(root, text='Correlation Coefficient:', bg='#473337',fg="white",font=11)
corr_label.place(x=50, y=550)

root.mainloop()