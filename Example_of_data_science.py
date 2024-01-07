# import require libraries 
import pandas as pd 
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# import data set 
music_data = pd.read_csv('music.csv')
# print(df.shape)   # describe - it will return count, mean , std, min 25% 50 % 75 % 

# II Cleaning - not needed 
# splitting for input and output sets 

X = music_data.drop(columns=['genre'])  # input
y = music_data['genre'] # output
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)  ## lower test_size then higher value for training data 

# learning and prediction 

model = DecisionTreeClassifier()
model.fit(X_train,y_train) 
predictions = model.predict(X_test) # we use X_test as there is a datasets for testing

# meassure accuracy of the model 

score = accuracy_score(y_test, predictions)
print(score)




