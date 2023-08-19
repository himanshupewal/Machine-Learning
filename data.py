import pandas as pd
import matplotlib.pylab as plt
from kmean import Kmeans

df = pd.read_csv('student_clustering.csv')

X= df.iloc[:,:].values

fm =   Kmeans(n_clusters=4,max_iter=100)
y_pred = fm.fit_predict(X)

plt.scatter(X[y_pred==0,0],X[y_pred==0,1])
plt.scatter(X[y_pred==1,0],X[y_pred==1,1])
plt.scatter(X[y_pred==2,0],X[y_pred==2,1])
plt.scatter(X[y_pred==3,0],X[y_pred==3,1])

plt.show()