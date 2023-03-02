import argparse
import numpy as np
import os
import pprint as pp
from concorde.tsp import TSPSolver
import time

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=500000)
    parser.add_argument("--num-nodes", type=int, default=50)
    parser.add_argument("--node-dim", type=int, default=2)
    parser.add_argument("--filename", type=str, default=None)
    opts=parser.parse_args()

    if opts.filename is None:
        opts.filename=f"tsp{opts.num_nodes}_concorde_v.txt"

    pp.pprint(vars(opts))
    start_time=time.time()
    nodes_set=np.random.uniform(0,1,(opts.num_samples, opts.num_nodes, opts.node_dim))
    with open(opts.filename, "w") as f:
        for nodes in nodes_set:
            solver=TSPSolver.from_data(nodes[:,0],nodes[:,1], norm="GEO")
            solution=solver.solve()
            path_coords=nodes[solution.tour]
            distance=np.sum(np.sqrt(np.sum(np.square(path_coords[1:]-path_coords[:-1]),axis=1)))
            cost=distance+np.sqrt(np.sum(np.square(path_coords[0]-path_coords[-1])))
            f.write(" ".join(str(x)+str(" ")+str(y) for x,y in nodes))
            f.write(str(" ")+str('path:')+str(" "))
            f.write(" ".join(str(x) for x in solution.tour))
            f.write(str(" ")+str('cost:')+str(" ")+str(cost)+str(" "))
            f.write("\n")
    
    print(f"Time taken: {time.time()-start_time}")




            

    

