from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

iris = load_iris()
X,y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)

for k in [1, 3, 5, 7]:
    print(f"Resultados para k = {k}:")
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_train,y_train)
    y_predict = clf.predict(X_test)
    print(classification_report(y_test, y_predict, target_names=iris.target_names))