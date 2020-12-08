#!/usr/bin/env python
# -*- coding: utf-8 -*-

#%%
import numpy as np
import argparse,os
import seaborn as sns

import matplotlib
matplotlib.use('Agg')

from plyfile import PlyData, PlyElement
import chainer
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainerui.utils import save_args

from chainer import training,datasets,iterators,Variable
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3
import matplotlib.colors as colors

from cosshift import CosineShift
from consts import optim,dtypes
from curvatureFlow import plot_trimesh2,plot_trimesh,compute_cos_angle_sub,compute_curvature_sub,compute_curvature_dmat_sub,cross,inprod,neighbour,save_ply

def plot_log(f,a,summary):
    a.set_yscale('log')

## updater 
class Updater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.coords = kwargs.pop('models')
        params = kwargs.pop('params')
        super(Updater, self).__init__(*args, **kwargs)
        self.args = params['args']
        self.faces = params['faces']
        self.N = params['N']
        self.fixed_coords = self.coords.W.array[self.args.fixed_vert]  # keep the fixed coords (note: deep copied)

    def update_core(self):
        opt = self.get_optimizer('opt')
        xp = self.coords.xp
        batch = self.get_iterator('main').next() ## inner edges
        e1, e2, d12 = self.converter(batch)
        vert = self.coords.W[:-1]
        beta = self.coords.W[-1,0]
        beta = 1
        L = F.sum( (vert[e1]-vert[e2])**2, axis=1 )
        loss = F.sum( (L-beta*d12)**2 )
        chainer.report({'loss': loss.item()}, self.coords)
        chainer.report({'beta': beta}, self.coords)
        # boundary
        if self.args.lambda_bdedge>0:
            ## edge length
            batch = self.get_iterator('bdedge').next()
            e1, e2, scale, weight = self.converter(batch)
            BL = F.sum((vert[e1]-vert[e2])**2, axis=1)
#            print(e1[:5],BL[:5],weight[:5],scale[:5],beta)
#            print(e1[-6:],BL[-6:],weight[-6:],scale[-6:],beta)
#            print((BL-scale*beta))
            loss_bd = F.sum( weight* ((BL-scale*beta)**2) )
            ## z-coords
            batch = self.get_iterator('bdvert').next()
            e1 = self.converter(batch)
            loss_bde += F.sum(vert[e1,2]**2)
            ## report
            chainer.report({'loss_bde': loss_bde.item()}, self.coords)
            loss = loss + self.args.lambda_bd * loss_bd
        if self.args.lambda_bdvert>0:
            #print(vert[self.args.fixed_vert])
            loss_bdv = F.sum( (self.fixed_coords-vert[self.args.fixed_vert])**2 )
            chainer.report({'loss_bdv': loss_bdv}, self.coords)
            loss = loss + self.args.lambda_bdvert * loss_bdv


        self.coords.cleargrads()
        loss.backward()
        opt.update(loss=loss)
        #self.coords.W.array[self.args.fixed_vert] = self.fixed_coords


        if (self.iteration) % self.args.vis_freq == 0 and self.args.vis_freq>0:
            # curvature
            K = compute_curvature_sub(vert,self.N,self.args.constrained_vert)
            loss_K = sum([abs(K[i]-self.args.targetK[i]) for i in range(len(K))])/len(K)
            chainer.report({'loss_K': loss_K}, self.coords)
            # plot
#            plot_trimesh(self.coords.W.array,self.faces,os.path.join(self.args.outdir,'count{:0>4}.jpg'.format(self.iteration)))



#%%
parser = argparse.ArgumentParser(description='Ranking learning')
parser.add_argument('--input', '-i', default="data/ex4b.ply", help='Path to ply file')
parser.add_argument('--output', default="output.ply", help='output ply filename')
parser.add_argument('--index_shift', type=int, default=1, help="vertex indices start at")
parser.add_argument('--target_curvature', '-K', default=None, type=str, help='file specifying target gaussian curvature')
parser.add_argument('--target_curvature_scalar', '-Ks', default=0.01, type=float, help='target gaussian curvature value')
parser.add_argument('--batchsize', '-b', type=int, default=-1,
                    help='Number of vertices which are updated at a time')
parser.add_argument('--epoch', '-e', type=int, default=2000,
                    help='Number of iterations')
parser.add_argument('--vis_freq', '-vf', type=int, default=20,
                    help='visualisation frequency')
parser.add_argument('--gpu', '-g', type=int, default=-1,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--outdir', '-o', default='result',
                    help='Directory to output the result')
parser.add_argument('--optimizer', '-op',choices=optim.keys(),default='Adam', help='optimizer')
parser.add_argument('--scaling', '-s', type=float, default=-1, help="scaling")
parser.add_argument('--lambda_bdedge', '-le', type=float, default=0, help="weight for boundary constraint")
parser.add_argument('--lambda_bdvert', '-lv', type=float, default=0, help="weight for boundary constraint")
parser.add_argument('--learning_rate', '-lr', type=float, default=1e-2, help='learning rate')
parser.add_argument('--verbose', '-v', action='store_true',help='print debug information')
args = parser.parse_args()

chainer.config.autotune = True
#chainer.print_runtime_info()
save_args(args, args.outdir)

# Read mesh data
fn, ext = os.path.splitext(args.input)
if (ext==".ply"):
    plydata = PlyData.read(args.input)
    vert = np.vstack([plydata['vertex']['x'],plydata['vertex']['y'],plydata['vertex']['z']]).astype(np.float64).T
    face = plydata['face']['vertex_indices']
else:
    vert = np.loadtxt(args.input)
    face = np.loadtxt(fn+"_face.txt").astype(np.uint16) -args.index_shift

dmat = np.loadtxt(fn+"_edgeLength.txt")  # target distance mat
inedge = np.loadtxt(fn+"_innerEdge.txt").astype(np.uint16) -args.index_shift
bdedge = np.loadtxt(fn+"_bounEdge.txt").astype(np.uint16) -args.index_shift
bdvert = np.loadtxt(fn+"_bounVertex.txt").astype(np.uint16) -args.index_shift
args.constrained_vert = np.loadtxt(fn+"_innerVertex.txt").astype(np.uint16) -args.index_shift
#np.savetxt("c.txt", args.constrained_vert,fmt='%i')
#np.savetxt("f.txt", bdvert,fmt='%i')

## initial scaling
if args.scaling>0:
    scaling = 1.0/args.scaling
else:
    scaling = np.max(vert)
vert /= scaling
# scaling for dmat
l1 = np.sum((vert[inedge[0,0]]-vert[inedge[0,1]])**2)
d1 = dmat[inedge[0,0],inedge[0,1]]**2
dmat *= np.sqrt(l1/d1)

# save scaled input mesh
save_ply(vert,face,os.path.join(args.outdir,'input.ply'))

# set target curvature
if args.target_curvature:
    args.targetK = np.loadtxt(args.target_curvature)
else:
    args.targetK = np.full(len(args.constrained_vert),args.target_curvature_scalar)
#    args.targetK = np.full((len(vert),),4*np.pi/len(vert)) ## constant curvature with euler char = 2

#%%
N = neighbour(len(vert),face)
args.fixed_vert = bdvert
#args.fixed_vert = np.zeros(1, dtype=np.uint16)
#%%
print("\nvertices {}, faces {}, fixed vertices {}, K-constrained {}".format(len(vert),len(face),len(args.fixed_vert),len(args.constrained_vert)))
ca = compute_curvature_sub(vert,N,args.constrained_vert,verbose=True)
print("\n\n Initial Curvature: ", [round(c.item(),5) for c in ca], "\n\n")
ca_ricci = compute_curvature_dmat_sub(dmat,N,args.constrained_vert)
print("\n\n Ricci flow Curvature: ", [round(c.item(),5) for c in ca_ricci], "\n\n")
#vert[args.fixed_vert]
#%%

if args.batchsize < 0:
    args.batchsize = len(inedge)

######################################

#beta = np.sum((vert[bdedge[0,0]]-vert[bdedge[0,1]])**2)
beta = 1.0
vert = np.concatenate([vert,np.array([[beta,0,0]])]) ## last entry is for scaling factor
coords = L.Parameter(vert)
opt = optim[args.optimizer](args.learning_rate)
opt.setup(coords)

## edge dataset
inedge_iter = chainer.iterators.SerialIterator(chainer.datasets.TupleDataset(inedge[:,0],inedge[:,1],dmat[inedge[:,0],inedge[:,1]]**2),args.batchsize,shuffle=False)
bdvert_iter = chainer.iterators.SerialIterator(chainer.datasets.TupleDataset(bdvert),len(bdvert),shuffle=False)

# TODO: quick-dirty boundary edge length constraint
S = np.ones(len(bdedge))
S[-4:] = 1/np.sqrt(2)
W = np.ones(len(bdedge))
E1 = np.concatenate([bdedge[:,0],np.array([1,1,180,180,1,27], dtype=np.uint16)-args.index_shift])
E2 = np.concatenate([bdedge[:,1],np.array([27,164,164,27,180,164], dtype=np.uint16)-args.index_shift])
S = np.concatenate([S, np.array([81,81,81,81,162,162])]).astype(np.float32)
W = np.concatenate([W, np.array([1,1,1,1,1,1])]).astype(np.float32)
bdedge_iter = chainer.iterators.SerialIterator(chainer.datasets.TupleDataset(E1,E2,S,W),len(S),shuffle=False)

#
if args.gpu >= 0:
    coords.to_gpu() 

updater = Updater(
    models=coords,
    iterator={'main':inedge_iter, 'bdedge':bdedge_iter, 'bdvert':bdvert_iter},
    optimizer={'opt': opt},
    device=args.gpu,
    params={'args': args, 'faces': face, 'N': N}
    )
trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.outdir)

log_interval = 10, 'iteration'
trainer.extend(extensions.LogReport(trigger=log_interval))
if extensions.PlotReport.available():
    trainer.extend(extensions.PlotReport(['opt/loss_K','opt/loss', 'opt/loss_bde','opt/loss_bdv'],'iteration', file_name='loss.png', postprocess=plot_log))
trainer.extend(extensions.PrintReport([
        'iteration', 'lr', 'opt/beta','opt/loss_K','opt/loss', 'opt/loss_bde','opt/loss_bdv','elapsed_time',
    ]),trigger=log_interval)
trainer.extend(extensions.ProgressBar(update_interval=1))
trainer.extend(extensions.observe_lr('opt'), trigger=log_interval)
trainer.extend(extensions.LogReport(trigger=log_interval))
## annealing
if args.optimizer in ['Adam','AdaBound','Eve']:
    lr_target = 'eta'
else:
    lr_target = 'lr'
trainer.extend(CosineShift(lr_target, args.epoch//2, optimizer=opt), trigger=(1, 'epoch'))
trainer.run()

####################################################
## result
if args.gpu >= 0:
    vert2 = coords.W.array.get()[:-1]
else:
    vert2 = coords.W.array[:-1]

ca_final = compute_curvature_sub(vert2,N,args.constrained_vert,verbose=True)
print("\n\n (final,target) Curvature: ", [(round(ca_final[i].item(),5),round(ca_ricci[i].item(),5)) for i in range(len(args.constrained_vert))])
print("boundary error: ", scaling*np.sqrt(np.sum( (vert[args.fixed_vert]-vert2[args.fixed_vert])**2 ) ))

plot_trimesh(vert2,face,os.path.join(args.outdir,"out.png"))
save_ply(vert2,face,os.path.join(args.outdir,args.output))
# graphs
n = len(ca)
sns.violinplot(x=np.array([0]*n), y=[c.item() for c in ca_ricci], cut=0)
plt.savefig(os.path.join(args.outdir,"curvature_target.png"))
plt.close()
sns.violinplot(x=np.array([0]*n), y=[c.item() for c in ca_final], cut=0)
plt.savefig(os.path.join(args.outdir,"curvature_final.png"))
plt.close()
sns.violinplot(x=np.array([0]*n), y=[abs(ca_ricci[i].item()-args.targetK[i])/args.targetK[i] for i in range(len(ca_ricci))], cut=0)
plt.savefig(os.path.join(args.outdir,"error_target.png"))
plt.close()
sns.violinplot(x=np.array([0]*n), y=[abs(ca_final[i].item()-ca_ricci[i].item())/ca_ricci[i].item() for i in range(len(ca_final))], cut=0)
plt.savefig(os.path.join(args.outdir,"error_final.png"))
plt.close()
