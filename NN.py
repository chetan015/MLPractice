from matplotlib import pyplot as plt

import numpy as np
data = [[19,4,17,17,2,1],
[17,3,18,2,20,2],
[1,19,0,17,5,20],
[4,20,2,1,18,18]
]
y = [[1,0,1,1,0,0],
[1,0,1,0,1,0],
[0,1,0,1,0,1],
[0,1,0,0,1,1],
]
mystery_student =[18,2,1,17,20,17]

#print(data[0])

for i in range(len(data)):
    point =data[i]
    color = "r"
    if point[2] ==0:
        color= "b"
    plt.scatter(point[0],point[1],c=color)


def sigmoid (x):
    return 1/(1 + np.exp(-x))
    
    
def sigmoid_p(x):
        return sigmoid(x)*(1-sigmoid(x))
    
  
T=np.linspace(-20,20,100)
print(T)
Y=sigmoid(T)
Z=sigmoid_p(T)
plt.plot(T,Y,c='r')
plt.plot(T,Z,c='b')
    
learning_rate=.2
w1=np.random.randn()
w2=np.random.randn()
w3=np.random.randn()
w4=np.random.randn()
w5=np.random.randn()
w6=np.random.randn()
b=np.random.randn()

costs=[]
for i in range(50000):
    ri =np.random.randint(len(data))
    #print(ri)
    point=data[ri]
 #   print (point)
#    print(point[0])
 #   print(point[1])
#    print(w2)
#    print(b)
    z=point[0] * w1 + 0.1 * point[1]* w2 + 0.2 * point[2]* w3 +0.15 * point[3]* w4 + 0.2 * point[4]* w5 + 0.1 * point[5]* w6+ b
 #   print(z)
    pred=sigmoid(z)
    target=Y[ri][0]
  
    cost=np.square(pred-target)
    costs.append(cost)
    if i%100==0:
        print(cost)
       

 #   print(point,cost)
     
    dcost_pred= 2*(pred-target)
    dpred_dz=sigmoid_p(z)
    
    dz_dw1=point[0]
    dz_dw2=point[1]
    dz_dw3=point[2]
    dz_dw4=point[3]
    dz_dw5=point[4]
    dz_dw6=point[5]
    dz_db=1
    
    dcost_dw1 =dcost_pred *dpred_dz*dz_dw1
    dcost_dw2 =dcost_pred *dpred_dz*dz_dw2
    dcost_dw3 =dcost_pred *dpred_dz*dz_dw3
    dcost_dw4 =dcost_pred *dpred_dz*dz_dw4
    dcost_dw5 =dcost_pred *dpred_dz*dz_dw5
    dcost_dw6 =dcost_pred *dpred_dz*dz_dw6
    dcost_db  =dcost_pred *dpred_dz*dz_db
    
    w1= w1 - learning_rate * dcost_dw1
    w2= w2 - learning_rate * dcost_dw2
    w3= w3 - learning_rate * dcost_dw3
    w4= w4 - learning_rate * dcost_dw4
    w5= w5 - learning_rate * dcost_dw5
    w6= w6 - learning_rate * dcost_dw6
    b = b - learning_rate * dcost_db
    
plt.plot(costs)

for i in range(len(data)):
    point=data[i]
    print(point)
    z=point[0] * w1 + 0.1 * point[1]* w2 + 0.2 * point[2]* w3 + 0.15 * point[3]* w4 + 0.2 * point[4]* w5 + 0.1 * point[5]* w6+ b
    pred=sigmoid(z)
    print("pred: {}".format(pred))