import numpy as np
x=np.ones(1,dtype='float32')

y=0.
finfo = np.finfo(x[0])
print (finfo.dtype, finfo.nexp, finfo.nmant)
m=0
while (y<1):
    x=x/2
    y=1.-x
    m=m+1
    print (y)
print (m-2)
