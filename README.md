# clarkpyopt
My optimisation tool box.

Just a collection of usefull classes i wrote to handle optimisation tasks. 


### Tiny demo
```
x = np.linspace(0, 10, 100)
y = np.sin(x) + np.random.randn(100) * .1

arrays = [(y * .5 + np.random.randn(100) * .1).reshape(-1, 1)-1,
          (y * 1.7 + np.random.randn(100) * .1).reshape(-1, 1)-1,
          (np.random.randn(100) * .5).reshape(-1, 1)]
X = np.concatenate(arrays, axis=1)
X = pd.DataFrame(X, columns=['A', 'B', 'C'])

model = Blend(fit_intercept=True, convex=True, lp_norm=1)
result = model.fit(X, y)
y_hat = model.predict(X)

# Print results
print('Coeff solution', result.weights)
print('Intercept', result.intercet)
print('Optimal value', result.value)
