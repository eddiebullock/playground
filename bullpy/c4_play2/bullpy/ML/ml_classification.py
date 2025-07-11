from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

#load good old fashioned iris
iris = load_iris()
x, y = iris.data, iris.target 

#split data 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

#train calssifier 
clf = RandomForestClassifier(random_state=42)
clf.fit(x_train, y_train)

#predict sweet sweet predictions 
y_pred = clf.predict(x_test)

#evaluate 
accuracy = accuracy_score(y_test, y_pred)
print(f"accuracy: {accuracy:.3f}")

#print classification report 
print(classification_report(y_test, y_pred, target_names=iris.target_names))