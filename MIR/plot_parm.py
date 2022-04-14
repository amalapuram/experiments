# import libraries
from matplotlib import pyplot as plt
import numpy as np
from numpy import dot
from numpy.linalg import norm
cosine_sim=[]
x_axis=[]


# Creating dataset
a = np.load("./param_weight_mem_100000__mini_batch_1024virtual_update.npy")
print("shape is:",a.shape)

# Creating plot
fig = plt.figure(figsize =(10, 7))
mean=np.mean(a[:10,:],axis=0)
a=a[10:,:]
print(a.shape)
weight=0.5
for i in range(len(a)):
    cos_sim = dot(mean, a[i,:])/(norm(mean)*norm(a[i,:]))
    mean=mean*weight+(1-weight)*a[i,:]
    x_axis.append(i)
    cosine_sim.append(cos_sim)


plt.plot(x_axis, cosine_sim)
plt.xlabel("Model parameters")  # add X-axis label
plt.ylabel("Cosine Similarity")  # add Y-axis label
#plt.title("Any suitable title")    #plt.hist(a[:,i])
plt.savefig("./param_weight_mem_100000__mini_batch_1024virtual_update.pdf")
