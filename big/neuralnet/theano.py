import theano
from theano import tensor as T

#initialize
x1 = T.scalar()
w1 = T.scalar()
w0 = T.scalar()
z1 = w1 * x1 + w0

#compile

net_input = theano.function(inputs = [w1, x1, w0], outputs = z1)

print("Net input: %.2f" % net_input(2.0, 1.0, 0.5))

print(theano.config.floatX)
### change to float32

theano.config.floatX = "float32"


### change default settings globally
# export THEANO_FLAGS=floatX=float32

# for a specific script
# THEANO_FLAGS=floatX=float32 python your_script.py

print(theano.config.device)


# execute a script from command line
# THEANO_FLAGS=device=cpu,floatX=float64 python your_script.py

# execute a script using GPU without making modifications to original code
# THEANO_FLAGS=device=gpu,floatX=float32 python your_script.py




import numpy as np

# initialize
x = T.fmatrix(name = "x")
x_sum = T.sum(x, axis = 0)

# compile
calc_sum = theano.function(inputs = [x], outputs = x_sum)

# execute (Python list)
ary = [[1, 2, 3], [1, 2, 3]]
print("Column sum:", calc_sum(ary))

# execute (Numpy array)
ary = np.array([[1, 2, 3], [1, 2, 3]], dtype = theano.config.floatX)
print("Column sum:", calc_sum(ary))

print(x.type())






# initialize
x = T.fmatrix("x")
w = theano.shared(np.asarray([[0.0, 0.0, 0.0]], dtype = theano.config.floatX))
z = x.dot(w.T)
update = [[w, w + 1.0]]


# compile
net_input = theano.function(inputs = [x], updates = update, outputs = z)

# execute
data = np.array([[1, 2, 3]], dtype = theano.config.floatX)
for i in range(5):
    print("z%d:" % i, net_input(data))
