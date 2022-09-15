import numpy as np
flow = np.load('sredjeni_flow.npy')
f= open("parovi.txt","w+")
width, height, _ = flow.shape
 
for v in range(height):
    for u in range(width):
            if (flow[u][v][2]>0.5):
                string = str(u)+' '+str(v)+' '+str(flow[u][v][0]+u)+' '+str(flow[u][v][1])+  '\n'
                f.write(string)
