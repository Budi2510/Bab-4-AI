from sklearn.neural_network import MLPClassifier

X = [
    [0.,0.,0.], [0.,0.,1.], [0.,1.,0.], [0.,1.,1.],
    [1.,0.,0.], [1.,0.,1.], [1.,1.,0.], [1.,1.,1.]
]
y = [0,0,0,0,0,0,0,1]

clf = MLPClassifier(
    solver="lbfgs",
    alpha=1e-5,
    hidden_layer_sizes=(5,2),
    random_state=1
)

clf.fit(X, y)

print("Prediksi:", clf.predict([
    [0,0,0],
    [1,1,1],
    [1,0,1]
]))

print("Bentuk bobot:", [coef.shape for coef in clf.coefs_])
