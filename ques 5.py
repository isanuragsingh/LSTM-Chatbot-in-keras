import numpy as np
import random
import matplotlib.pyplot as plt
m=[]
for i in range(0,4096):
    A=0
    a=np.random.randint(0,(2**14),(2**15))
    x=np.array(a,dtype='int8')
    i=0
    A=np.sum(x)
    m.append(A)

plt.hist(m,100)
plt.show()


