import numpy as np

weights = np.asarray([
[0.0,0.0,0.0,0.0,0.0
,0.0,0.0,0.0,0.0,0.0
,0.0,0.0,-1.0,1.0,0.0
,0.0,0.0,0.0,0.0,0.0
,0.0,0.0,0.0,0.0,0.0],



[0.0,0.0,0.0,0.0,0.0
,0.0,0.0,0.0,0.0,0.0
,0.0,0.0,-1.0,0.0,0.0
,0.0,0.0,0.0,1.0,0.0
,0.0,0.0,0.0,0.0,0.0],

[ 0.0,0.0,0.0,0.0,0.0
,0.0,0.0,0.0,0.0,0.0
,0.0,0.0,-1.0,0.0,0.0
,0.0,0.0,1.0,0.0,0.0
,0.0,0.0,0.0,0.0,0.0],


[0.0,0.0,0.0,0.0,0.0
,0.0,0.0,0.0,0.0,0.0
,0.0,0.0,-1.0,0.0,0.0
,0.0,1.0,0.0,0.0,0.0
,0.0,0.0,0.0,0.0,0.0],


[0.0,0.0,0.0,0.0,0.0
,0.0,0.0,0.0,0.0,0.0
,0.0,1.0,-1.0,0.0,0.0
,0.0,0.0,0.0,0.0,0.0
,0.0,0.0,0.0,0.0,0.0],


[0.0,0.0,0.0,0.0,0.0
,0.0,1.0,0.0,0.0,0.0
,0.0,0.0,-1.0,0.0,0.0
,0.0,0.0,0.0,0.0,0.0
,0.0,0.0,0.0,0.0,0.0],


[0.0,0.0,0.0,0.0,0.0
,0.0,0.0,1.0,0.0,0.0
,0.0,0.0,-1.0,0.0,0.0
,0.0,0.0,0.0,0.0,0.0
,0.0,0.0,0.0,0.0,0.0],


[0.0,0.0,0.0,0.0,0.0
,0.0,0.0,0.0,1.0,0.0
,0.0,0.0,-1.0,0.0,0.0
,0.0,0.0,0.0,0.0,0.0
,0.0,0.0,0.0,0.0,0.0],

[0.0,0.0,0.0,0.0,0.0
,0.0,0.0,0.0,0.0,0.0
,0.0,1.0,-2.0,1.0,0.0
,0.0,0.0,0.0,0.0,0.0
,0.0,0.0,0.0,0.0,0.0],


 [0.0,0.0,0.0,0.0,0.0
,0.0,1.0,0.0,0.0,0.0
,0.0,0.0,-2.0,0.0,0.0
,0.0,0.0,0.0,1.0,0.0
,0.0,0.0,0.0,0.0,0.0],



 [0.0,0.0,0.0,0.0,0.0
,0.0,0.0,1.0,0.0,0.0
,0.0,0.0,-2.0,0.0,0.0
,0.0,0.0,1.0,0.0,0.0
,0.0,0.0,0.0,0.0,0.0],



 [0.0,0.0,0.0,0.0,0.0
,0.0,0.0,0.0,1.0,0.0
,0.0,0.0,-2.0,0.0,0.0
,0.0,1.0,0.0,0.0,0.0
,0.0,0.0,0.0,0.0,0.0],



 [0.0,0.0,0.0,0.0,0.0
,0.0,0.0,0.0,0.0,0.0
,0.0,1.0,-3.0,1.0,1.0
,0.0,0.0,0.0,0.0,0.0
,0.0,0.0,0.0,0.0,0.0],



 [0.0,0.0,0.0,0.0,0.0
,0.0,1.0,0.0,0.0,0.0
,0.0,0.0,-3.0,0.0,0.0
,0.0,0.0,0.0,1.0,0.0
,0.0,0.0,0.0,0.0,1.0],




 [0.0,0.0,0.0,0.0,0.0
,0.0,0.0,1.0,0.0,0.0
,0.0,0.0,-3.0,0.0,0.0
,0.0,0.0,1.0,0.0,0.0
,0.0,0.0,1.0,0.0,0.0],



 [0.0,0.0,0.0,0.0,0.0
,0.0,0.0,0.0,1.0,0.0
,0.0,0.0,-3.0,0.0,0.0
,0.0,1.0,0.0,0.0,0.0
,1.0,0.0,0.0,0.0,0.0],


 [0.0,0.0,0.0,0.0,0.0
,0.0,0.0,0.0,0.0,0.0
,1.0,1.0,-3.0,1.0,0.0
,0.0,0.0,0.0,0.0,0.0
,0.0,0.0,0.0,0.0,0.0],




 [1.0,0.0,0.0,0.0,0.0
,0.0,1.0,0.0,0.0,0.0
,0.0,0.0,-3.0,0.0,0.0
,0.0,0.0,0.0,1.0,0.0
,0.0,0.0,0.0,0.0,0.0],




 [0.0,0.0,1.0,0.0,0.0
,0.0,0.0,1.0,0.0,0.0
,0.0,0.0,-3.0,0.0,0.0
,0.0,0.0,1.0,0.0,0.0
,0.0,0.0,0.0,0.0,0.0],



 [0.0,0.0,0.0,0.0,1.0
,0.0,0.0,0.0,1.0,0.0
,0.0,0.0,-3.0,0.0,0.0
,0.0,1.0,0.0,0.0,0.0
,0.0,0.0,0.0,0.0,0.0],


 [0.0,0.0,0.0,0.0,0.0
,0.0,-1.0,2.0,-1.0,0.0
,0.0,2.0,-4.0,2.0,0.0
,0.0,-1.0,2.0,-1.0,0.0
,0.0,0.0,0.0,0.0,0.0],



 [0.0,0.0,0.0,0.0,0.0
,0.0,-1.0,2.0,-1.0,0.0
,0.0,2.0,-4.0,2.0,0.0
,0.0,0.0,0.0,0.0,0.0
,0.0,0.0,0.0,0.0,0.0],



 [0.0,0.0,0.0,0.0,0.0
,0.0,-1.0,2.0,0.0,0.0
,0.0,2.0,-4.0,0.0,0.0
,0.0,-1.0,2.0,0.0,0.0
,0.0,0.0,0.0,0.0,0.0],




 [0.0,0.0,0.0,0.0,0.0
,0.0,0.0,0.0,0.0,0.0
,0.0,2.0,-4.0,2.0,0.0
,0.0,-1.0,2.0,-1.0,0.0
,0.0,0.0,0.0,0.0,0.0],




 [0.0,0.0,0.0,0.0,0.0
,0.0,0.0,2.0,-1.0,0.0
,0.0,0.0,-4.0,2.0,0.0
,0.0,0.0,2.0,-1.0,0.0
,0.0,0.0,0.0,0.0,0.0],



 [-1.0,2.0,-2.0,2.0,-1.0
,2.0,-6.0,8.0,-6.0,2.0
,-2.0,8.0,-12.0,8.0,-2.0
,2.0,-6.0,8.0,-6.0,2.0
,-1.0,2.0,-2.0,2.0,-1.0],

 [-1.0,2.0,-2.0,2.0,-1.0
,2.0,-6.0,8.0,-6.0,2.0
,-2.0,8.0,-12.0,8.0,-2.0
,0.0,0.0,0.0,0.0,0.0
,0.0,0.0,0.0,0.0,0.0],


 [0.0,0.0,-2.0,2.0,-1.0
,0.0,0.0,8.0,-6.0,2.0
,0.0,0.0,-12.0,8.0,-2.0
,0.0,0.0,8.0,-6.0,2.0
,0.0,0.0,-2.0,2.0,-1.0],


 [0.0,0.0,0.0,0.0,0.0
,0.0,0.0,0.0,0.0,0.0
,-2.0,8.0,-12.0,8.0,-2.0
,2.0,-6.0,8.0,-6.0,2.0
,-1.0,2.0,-2.0,2.0,-1.0],


 [-1.0,2.0,-2.0,0.0,0.0
   ,2.0,-6.0,8.0,0.0,0.0
   ,-2.0,8.0,-12.0,0.0,0.0
   ,2.0,-6.0,8.0,0.0,0.0
   ,-1.0,2.0,-2.0,0.0,0.0]])
 

weights = weights.T.reshape((5,5,30))
weights = np.expand_dims(weights, axis=2)



hp_filters = [weights, np.asarray([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])]
