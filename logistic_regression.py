import numpy as np 
import theano as th 
import theano.tensor as T 
rng=np.random

N= 400
feats=784

D=(rng.randn(N, feats), rng.randint(size=N, low=0, high=2))
training_steps=10000

x= T.dmatrix("x")
y= T.dvector("y")

#weights & bias
w= th.shared(rng.randn(feats), name="w")
b= th.shared( 0., name="b")

print ("initial model")
print (w.get_value())
print (b.get_value())

p1= 1/(1+T.exp(-T.dot(x,w)-b))
prediction= p1>0.5
xent= -y*T.log(p1)-(1-y)*T.log(1-p1)
cost= xent.mean()+ 0.01*(w**2).sum()
gw, gb= T.grad(cost, [w,b])

train = th.function(
          inputs=[x,y],
          outputs=[prediction, xent],
          updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb)))
predict= th.function(inputs= [x], outputs=prediction)

for i in range(training_steps):
	pred, err= train(D[0], D[1])


print("Final model:")
print(w.get_value())
print(b.get_value())
print("target values for D:")
print(D[1])
print("prediction on D:")
print(predict(D[0]))
