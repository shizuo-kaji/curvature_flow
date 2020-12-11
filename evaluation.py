#!/usr/bin/env python
## pip install autograd
import argparse,os,time
import seaborn as sns
import autograd.numpy as np
from autograd import grad, jacobian, hessian
from scipy.optimize import minimize,NonlinearConstraint,LinearConstraint,least_squares
import matplotlib.pyplot as plt
from curvatureFlow import plot_trimesh2,plot_trimesh,compute_cos_angle_sub,compute_curvature_sub,compute_curvature_dmat_sub,cross,inprod,neighbour,save_ply
from plyfile import PlyData, PlyElement


#########################
parser = argparse.ArgumentParser(description='embedding of metric graphs')
parser.add_argument('--input', '-i', default="data/Ex4b_final.ply", help='prefix')
parser.add_argument('--boundary_vertex', '-bv', default="data/Ex4b_boundary.csv", help='Path to a csv specifying boundary position')
parser.add_argument('--constrained_vert', '-cv', default="data/Ex4b_cv.txt", type=str, help='file containing indices of vertices with curvature target')
parser.add_argument('--edge_length', '-el', default="data/Ex4b_edge_scaled.csv", help='Path to a csv specifying edge length')
parser.add_argument('--index_shift', type=int, default=1, help="vertex indices start at")
parser.add_argument('--outdir', '-o', default='result',help='Directory to output the result')
parser.add_argument('--target_curvature', '-K', default=None, type=str, help='file specifying target gaussian curvature')
parser.add_argument('--target_curvature_scalar', '-Ks', default=0.01, type=float, help='target gaussian curvature value')
args = parser.parse_args()


# Read mesh data
fn, ext = os.path.splitext(args.input)
fn = fn.rsplit('_', 1)[0]
if (ext==".ply"):
    plydata = PlyData.read(args.input)
    vert = np.vstack([plydata['vertex']['x'],plydata['vertex']['y'],plydata['vertex']['z']]).astype(np.float64).T
    face = plydata['face']['vertex_indices']
    edgedat = np.loadtxt(args.edge_length,delimiter=",")
    inedge = edgedat[:,:2].astype(np.uint32)
    edgelen = edgedat[:,2]
    dmat = np.sum((np.expand_dims(vert,axis=0) - np.expand_dims(vert,axis=1))**2,axis=2)
    dmat[inedge[:,0],inedge[:,1]] = edgelen
    dmat[inedge[:,1],inedge[:,0]] = edgelen
#    np.savetxt("dmat.txt",dmat)
    if args.boundary_vertex:
        bddat = np.loadtxt(args.boundary_vertex,delimiter=",")
        args.fixed_vert = bddat[:,0].astype(np.uint32)
        fixed_coords = bddat[:,1:]
    else:
        args.fixed_vert =np.array(0)
        fixed_coords = np.array(0)

else:  # old style input data format
    index_shift = 1
    vert = np.loadtxt(args.input)
    face = np.loadtxt(fn+"_face.txt").astype(np.uint16) -index_shift
    dmat = np.loadtxt(fn+"_edgeLength_scaled.txt")  # target distance mat
    inedge = np.loadtxt(fn+"_innerEdge.txt").astype(np.uint16) -index_shift
    args.fixed_vert = np.loadtxt(fn+"_bounVertex.txt").astype(np.uint16) -index_shift
    vert_init = np.loadtxt(fn+"_init.txt")
    fixed_coords = vert_init[args.fixed_vert]
    edgelen = dmat[inedge[:,0],inedge[:,1]]
    np.savetxt(fn+"_edge.csv", np.hstack([inedge,edgelen[:,np.newaxis]]),delimiter=",")
    np.savetxt(fn+"_boundary.csv", np.hstack([args.fixed_vert[:,np.newaxis],fixed_coords]),delimiter=",")

#
args.free_vert = list(set(range(len(vert))) - set(args.fixed_vert))
if args.constrained_vert:
    args.constrained_vert = np.loadtxt(args.constrained_vert).astype(np.uint16)
else:
    args.constrained_vert = list(set(args.free_vert) - set(np.where( args.target_curvature == -99 )[0]))

N = neighbour(len(vert),face)
print("\nvertices {}, faces {}, fixed vertices {}, K-constrained {}".format(len(vert),len(face),len(args.fixed_vert),len(args.constrained_vert)))

# set target curvature
if args.target_curvature:
    args.targetK = np.loadtxt(args.target_curvature)
else:
    args.targetK = np.full(len(args.constrained_vert),args.target_curvature_scalar)

# %%
ca_final = compute_curvature_sub(vert,N,args.constrained_vert,verbose=True)
ca_dmat = compute_curvature_dmat_sub(dmat,N,args.constrained_vert)
print("\n\n (final,dmat,target) Curvature: ", [(round(ca_final[i].item(),5),round(ca_dmat[i].item(),5),round(args.targetK[i],5)) for i in range(len(args.constrained_vert))])
bd_error = ( (fixed_coords-vert[args.fixed_vert])**2 )
l2 = np.sum( (vert[inedge[:,0]]-vert[inedge[:,1]])**2, axis=1 )
edge_error = (l2-dmat[inedge[:,0],inedge[:,1]]**2)**2
print("edge^2 squared error: {}, boundary squared error: {}".format(np.sum(edge_error),np.sum(bd_error)))

# graphs
sns.violinplot(y=edge_error, cut=0)
plt.savefig(os.path.join(args.outdir,"edge_error.png"))
plt.close()
sns.violinplot(y=bd_error, cut=0)
plt.savefig(os.path.join(args.outdir,"boundary_error.png"))
plt.close()

n = len(args.constrained_vert)
sns.violinplot(x=np.array([0]*n), y=args.targetK, cut=0)
plt.savefig(os.path.join(args.outdir,"curvature_target.png"))
plt.close()
sns.violinplot(x=np.array([0]*n), y=[c.item() for c in ca_dmat], cut=0)
plt.savefig(os.path.join(args.outdir,"curvature_dmat.png"))
plt.close()
sns.violinplot(x=np.array([0]*n), y=[c.item() for c in ca_final], cut=0)
plt.savefig(os.path.join(args.outdir,"curvature_final.png"))
plt.close()
sns.violinplot(x=np.array([0]*n), y=[abs(ca_dmat[i].item()-args.targetK[i]) for i in range(len(ca_dmat))], cut=0)
plt.savefig(os.path.join(args.outdir,"error_dmat.png"))
plt.close()
sns.violinplot(x=np.array([0]*n), y=[abs(ca_final[i].item()-ca_dmat[i].item()) for i in range(len(ca_final))], cut=0)
plt.savefig(os.path.join(args.outdir,"error_final_vs_dmat.png"))
plt.close()
sns.violinplot(x=np.array([0]*n), y=[abs(ca_final[i].item()-args.targetK[i]) for i in range(len(ca_final))], cut=0)
plt.savefig(os.path.join(args.outdir,"error_final.png"))
plt.close()