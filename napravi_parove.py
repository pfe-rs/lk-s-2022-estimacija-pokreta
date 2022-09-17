import numpy as np
flow = np.load('sredjeni_flow.npy')
f= open("parovi.txt","w+")
height, width, _ = flow.shape
print(flow.shape) 


for v in range(height):
    for u in range(width):
            if (flow[v][u][2]>0.5):
                string = str(u)+' '+str(v)+' '+str(flow[v][u][0]+u)+' '+str(flow[v][u][1]+v)+  '\n'
                f.write(string)
