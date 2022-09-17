import numpy as np

def parovi(npyfile, txtfile):
    flow = np.load(npyfile)
    f= open(txtfile,"w+")
    height, width, _ = flow.shape


    for v in range(height):
        for u in range(width):
                if (flow[v][u][2]>0.5):
                    string = str(u)+' '+str(v)+' '+str(flow[v][u][0]+u)+' '+str(flow[v][u][1]+v)+  '\n'
                    f.write(string)
