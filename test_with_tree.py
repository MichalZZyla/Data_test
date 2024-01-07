import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

music_data = pd.read_csv('music.csv')
X = music_data.drop(columns=['genre'])  # input
y = music_data['genre'] # output

model = DecisionTreeClassifier()
model.fit(X,y)

tree.export_graphviz(model, out_file= 'music-recommender.dot',
                     feature_names = ['age','gender'],
                     class_names =sorted(y.unique()),
                     label = 'all',  # every box has label 
                     rounded = True,  # rounded boxes
                     filled= True)   # rach box is filled with the color 