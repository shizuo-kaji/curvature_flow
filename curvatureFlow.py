#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import argparse,os

import matplotlib
matplotlib.use('Agg')
import seaborn as sns

from plyfile import PlyData, PlyElement
import chainer
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer import training,datasets,iterators,Variable
from consts import optim,dtypes
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3
import matplotlib.colors as colors
from cosshift import CosineShift

def plot_log(f,a,summary):
    a.set_yscale('log')

## triangle mesh plot
def plot_trimesh2(vert,tri,fname):
    m = np.min(verts)
    M = np.max(verts)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    #ax = fig.gca(projection='3d')
    ax.set_xlim3d=(m-0.2*(M-m),M+0.2*(M-m))
    ax.set_ylim3d=(m-0.2*(M-m),M+0.2*(M-m))
    ax.set_zlim3d=(m-0.2*(M-m),M+0.2*(M-m))
    ax.set_axis_off()
    ax.plot_trisurf(vert[:,0], vert[:,1], vert[:,2], triangles=tri, linewidth=0.2, antialiased=True, cmap=plt.cm.cividis)
#    plt.show()
    plt.savefig(fname, dpi=200)
    plt.close()

def plot_trimesh(verts,faces,fname):
    m = np.min(verts)
    M = np.max(verts)
    ax = a3.Axes3D(plt.figure())
    ax.dist=8
    ax.azim=-140
    ax.elev=50
    ax.set_axis_off()
    ax.set_xlim([m-0.2*(M-m),M+0.2*(M-m)])
    ax.set_ylim([m-0.2*(M-m),M+0.2*(M-m)])
    ax.set_zlim([m-0.2*(M-m),M+0.2*(M-m)])
    for f in faces:
        triangle=[verts[f[0]],verts[f[1]],verts[f[2]]] 
        face = a3.art3d.Poly3DCollection([triangle]) 
#        face.set_color(colors.rgb2hex(np.random.random(3)))
        face.set_edgecolor('k')
#        face.set_alpha(0.9)
        ax.add_collection3d(face)
    plt.savefig(fname, dpi=200)
    plt.close()
#    plt.show()


# Compute cosine of angle defect
def compute_cos_angle(vert,face,theta,xp):
    # Obsolite: replaced with compute_cos_angle_sub which is more efficient
    cos_angle = Variable(xp.cos(theta))
    sin_angle = Variable(xp.sin(theta))
    for f in face:
        n = len(f)
        id_p = xp.array([f[(i-1)%n] for i in range(n+1)])
        id = xp.array([f[i%n] for i in range(n+1)])
        id_n = xp.array([f[(i+1)%n] for i in range(n)])
        L = F.sum((vert[id_p] - vert[id])**2, axis=1)
        D = F.sum((vert[id_n] - vert[ id_p[:-1] ])**2, axis=1)
        c1 = (L[:n]+L[1:]-D)/(2*F.sqrt(L[:n]*L[1:]))
        s1 = F.sqrt(1-c1**2)
        # trigonometric addition formula
        c0 = cos_angle[f]
        s0 = sin_angle[f]
        cos_angle = F.scatter_add(cos_angle,f,-c0 + c0*c1 - s0*s1)
        sin_angle = F.scatter_add(sin_angle,f,-s0 + c0*s1 + s0*c1)
    return cos_angle

# Compute cosine of angle defect for vertex with indices idx
def compute_cos_angle_sub(vert,N,theta,idx,xp):
    cos_angle = []
    for i in idx:
        L0 = F.sum((vert[N[i][:,0]] - vert[i])**2, axis=1)
        L1 = F.sum((vert[N[i][:,1]] - vert[i])**2, axis=1)
        D = F.sum((vert[N[i][:,1]] - vert[N[i][:,0]])**2, axis=1)
        c1 = (L0+L1-D)/(2*F.sqrt(L0*L1)) # law of cosines
        s1 = F.sqrt(1-c1**2)
#        print(xp.arccos(c1.array),xp.arcsin(s1.array))
        c0,s0 = xp.cos(theta[i]),xp.sin(theta[i])
        for j in range(len(c1)): # addition law
            c0,s0 = c0*c1[j]-s0*s1[j], c0*s1[j]+s0*c1[j]    # don't split (or you need a temporary variable)
        cos_angle.append(c0)
    return(cos_angle)

# Compute gaussian curvature
def compute_curvature(vert,face,xp):
    # Obsolite: replaced with compute_curvature_sub which is more efficient
    K = Variable(xp.full((len(vert),), 2*xp.pi))
    for f in face:
        n = len(f)
        id_p = xp.array([f[(i-1)%n] for i in range(n+1)])
        id = xp.array([f[i%n] for i in range(n+1)])
        id_n = xp.array([f[(i+1)%n] for i in range(n)])
        L = F.sum((vert[id_p] - vert[id])**2, axis=1)
        D = F.sum((vert[id_n] - vert[ id_p[:-1] ])**2, axis=1)
        c1 = (L[:n]+L[1:]-D)/(2*F.sqrt(L[:n]*L[1:]))
        K = F.scatter_add(K,f,-F.arccos(c1))
    return K

# Compute gaussian curvature for vertex with indices idx
def compute_curvature_sub(vert,N,idx,force_upward=False,verbose=False):
    K = []
    for i in idx:
        L0 = F.sum((vert[N[i][:,0]] - vert[i])**2, axis=1)
        L1 = F.sum((vert[N[i][:,1]] - vert[i])**2, axis=1)
        D = F.sum((vert[N[i][:,1]] - vert[N[i][:,0]])**2, axis=1)
        c1 = (L0+L1-D)/(2*F.sqrt(L0*L1))
        arg = 2*np.pi-F.sum(F.arccos(c1))
        if force_upward:
            up = F.sum(vert[i]-vert[N[i][:,0]],axis=0)
            fn = [0,0,0]
            for k in range(len(N[i])):
                q = cross(vert[N[i][k,1]] - vert[i], vert[N[i][k,0]] - vert[i])
                fn[0] += q[0]
                fn[1] += q[1]
                fn[2] += q[2]
            s = F.sign(inprod(fn,up))
            if verbose and s.array<0:
                print(i,s)
            arg *= F.sign(inprod(fn,up))
        K.append(arg)
    return(K)

# Compute gaussian curvature for vertex with indices idx from distance matrix
def compute_curvature_dmat_sub(dmat,N,idx):
    K = []
    for i in idx:
        L0 = dmat[N[i][:,0],i]**2
        L1 = dmat[N[i][:,1],i]**2
        D = dmat[N[i][:,0],N[i][:,1]]**2
        c1 = (L0+L1-D)/(2*F.sqrt(L0*L1))
        arg = 2*np.pi-F.sum(F.arccos(c1))
        K.append(arg)
    return(K)

## create ply file
def save_ply(vert,face,fname):
    el1 = PlyElement.describe(np.array([(x[0],x[1],x[2]) for x in vert],dtype=[('x', 'f8'), ('y', 'f8'),('z', 'f8')]), 'vertex')
    el2 = PlyElement.describe(np.array([([x[0],x[1],x[2]], 0) for x in face],dtype=[('vertex_indices', 'i4', (3,)), ('red', 'u1')]), 'face')
    PlyData([el1,el2], text=True).write(fname)

def cross(p,q):
    return( (p[1]*q[2]-p[2]*q[1],p[2]*q[0]-p[0]*q[2], p[0]*q[1]-p[1]*q[0]))

def inprod(p,q):
    return( p[0]*q[0]+p[1]*q[1]+p[2]*q[2])

# vertex star
# for a vertex i: N[i][k,0], i, N[i][k,1] form consecutive edges
def neighbour(n,face):
    F = [[] for i in range(n)]
    for f in face:
        for i in range(len(f)-1):
            F[f[i]].append([f[i-1],f[i+1]])
        F[f[-1]].append([f[-2],f[0]])
    return([np.array(F[i]) for i in range(n)])

######################################################################
## updater 
class Updater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.coords = kwargs.pop('models')
        params = kwargs.pop('params')
        super(Updater, self).__init__(*args, **kwargs)
        self.args = params['args']
        self.faces = params['faces']
        self.N = params['N']
        self.fixed_coords = params['fixed_coords']
        self.force_upward = False

    def update_core(self):
        opt = self.get_optimizer('opt')
        xp = self.coords.xp
        b = self.get_iterator('main').next()
        vert = self.coords.W
        if self.args.optimise_cos:
#            ca = compute_cos_angle(self.coords.W,self.faces,self.args.target_curvature,xp)
#            loss =  sum([1-ca[i] for i in self.args.constrained_vert])   #/len(self.args.free_vert)
            ca = compute_cos_angle_sub(vert,self.N,self.args.target_curvature,b,xp)
            loss =  sum([1-ca[i] for i in range(len(ca))])/len(b)
        else:
#            K = compute_curvature(vert,self.faces,xp)
#            loss = sum([(K[i]-self.args.target_curvature[i])**2 for i in self.args.constrained_vert])
            K = compute_curvature_sub(vert,self.N,b)
            loss = sum([(K[i]-self.args.target_curvature[b[i]])**2 for i in range(len(b))])/len(b)
#            print([(i,K[i],self.args.target_curvature[b[i]]) for i in range(len(b))])
        chainer.report({'loss': loss.item()}, self.coords)

        if self.force_upward: # each vertex should be higher in z direction than the average of neighbours
            for i in b:
                up = F.sum(vert[i]-vert[self.N[i][:,0]],axis=0)
                fn = [0,0,0]
                for k in range(len(self.N[i])):
                    q = cross(vert[self.N[i][k,1]] - vert[i], vert[self.N[i][k,0]] - vert[i])
                    fn[0] += q[0]
                    fn[1] += q[1]
                    fn[2] += q[2]
                loss_upward = F.relu(-inprod(fn,up)-0.1)**2
            chainer.report({'loss_up': loss_upward}, self.coords)
            loss += self.args.lambda_upward * loss_upward

        if self.args.lambda_bdvert>0:
            loss_bdv = F.sum( (self.fixed_coords-vert[self.args.fixed_vert])**2 )
            chainer.report({'loss_bdv': loss_bdv}, self.coords)
            loss += self.args.lambda_bdvert * loss_bdv

        self.coords.cleargrads()
        loss.backward()
        opt.update(loss=loss)

        if self.args.strict_boundary:
            self.coords.W.array[self.args.fixed_vert] = self.fixed_coords

        if (self.iteration) % self.args.vis_freq == 0 and self.args.vis_freq>0:
            plot_trimesh(self.coords.W.array,self.faces,os.path.join(self.args.outdir,'count{:0>4}.jpg'.format(self.iteration)))

#####################################################################################
#-----------------------
def main():
    parser = argparse.ArgumentParser(description='Ranking learning')
    parser.add_argument('input', help='Path to ply file')
    parser.add_argument('--output', default="output.ply", help='output ply filename')
    parser.add_argument('--target_curvature', '-K', default=None, type=str, help='file containing target gaussian curvature')
    parser.add_argument('--target_curvature_scalar', '-Ks', default=0.01, type=float, help='target gaussian curvature value')
    parser.add_argument('--constrained_vert', '-cv', default=None, type=str, help='file containing indices of vertices with curvature target')
    parser.add_argument('--boundary_vertex', '-bv', default=None, help='Path to a csv specifying boundary position')
    parser.add_argument('--lambda_bdvert', '-lv', type=float, default=0, help="weight for boundary constraint")
    parser.add_argument('--strict_boundary', '-sbd', action='store_true',help='strict boundary constraint')
    parser.add_argument('--lambda_upward', '-lu', type=float, default=0, help="weight for upwardness")
    parser.add_argument('--batchsize', '-b', type=int, default=-1,
                        help='Number of vertices which are updated at a time')
    parser.add_argument('--epoch', '-e', type=int, default=100,
                        help='Number of iterations')
    parser.add_argument('--vis_freq', '-vf', type=int, default=-1,
                        help='visualisation frequency')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--optimise_cos', '-cos', action='store_true', help='optimise cosine rather than curvature')
    parser.add_argument('--outdir', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--optimizer', '-op',choices=optim.keys(),default='Adam',
                        help='optimizer')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--salt', action='store_true',help='add salt to randomise initial coordinates')
    parser.add_argument('--verbose', '-v', action='store_true',help='print debug information')
    args = parser.parse_args()

    chainer.config.autotune = True
    #chainer.print_runtime_info()

    # Read mesh data
    plydata = PlyData.read(args.input)
    vert = np.vstack([plydata['vertex']['x'],plydata['vertex']['y'],plydata['vertex']['z']]).astype(np.float64).T
    face = plydata['face']['vertex_indices']
    print(args)
#    plot_trimesh(vert,face,"out.png")

    # set target curvature
    if args.target_curvature:
        args.target_curvature = np.loadtxt(args.target_curvature)
    else:
#        args.target_curvature = np.full((len(vert),),4*np.pi/len(vert)) ## constant curvature with euler char = 2
        args.target_curvature = np.full((len(vert),),args.target_curvature_scalar) ## constant curvature with euler char = 2
    if len(args.target_curvature) != len(vert):
        print("Curvatures and vertices have different length!")
        exit(-1)

    # determine fixed vertices
    args.vert = range(len(vert))
    if args.boundary_vertex:
        bddat = np.loadtxt(args.boundary_vertex,delimiter=",")
        args.fixed_vert = bddat[:,0].astype(np.uint32)
        fixed_coords = bddat[:,1:]
    else:
        args.fixed_vert = np.where( args.target_curvature > 2*np.pi )[0]
        fixed_coords = vert[args.fixed_vert]
#        np.savetxt("boundary.csv", np.hstack([args.fixed_vert[:,np.newaxis],fixed_coords]),delimiter=",",fmt='%i,%f,%f,%f')


    args.free_vert = list(set(args.vert) - set(args.fixed_vert))
    if args.constrained_vert:
        args.constrained_vert = np.loadtxt(args.constrained_vert).astype(np.uint16)
    else:
        args.constrained_vert = list(set(args.free_vert) - set(np.where( args.target_curvature == -99 )[0]))
#        np.savetxt("cv.txt", args.constrained_vert, fmt='%i')
    if args.salt:
        vert[args.free_vert] += np.random.randn(*vert[args.free_vert].shape)*1e-4


    print("\nvertices {}, faces {}, fixed vertices {}, vertices with target curvature {}".format(len(vert),len(face),len(args.fixed_vert),len(args.constrained_vert)))
    if args.batchsize < 0:
        args.batchsize = (len(args.constrained_vert)+1) //2
    ######################################
    N = neighbour(len(vert),face)
#        ca = compute_curvature(vert,face,np)
    ca = compute_curvature_sub(vert,N,args.constrained_vert,verbose=True)
#        ca = compute_cos_angle_sub(vert,N,np.zeros(len(vert)),args.vert,np)
    print("\n\n Initial Curvature: ", [round(c.item(),5) for c in ca], "\n\n")

    coords = L.Parameter(vert)
    opt = optim[args.optimizer](args.learning_rate)
    opt.setup(coords)
    id_iter = chainer.iterators.SerialIterator(chainer.dataset.tabular.from_data(args.constrained_vert),args.batchsize)

    if args.gpu >= 0:
        coords.to_gpu() 

    updater = Updater(
        models=coords,
        iterator=id_iter,
        optimizer={'opt': opt},
        device=args.gpu,
        params={'args': args, 'faces': face, 'N': N, 'fixed_coords': fixed_coords}
        )
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.outdir)

    log_interval = 5, 'iteration'
    trainer.extend(extensions.LogReport(trigger=log_interval))
    if extensions.PlotReport.available():
        trainer.extend(extensions.PlotReport(['opt/loss', 'opt/loss_bdv', 'opt/loss_up'],'iteration', file_name='loss.png', postprocess=plot_log))
    trainer.extend(extensions.PrintReport([
            'iteration', 'lr', 'opt/loss', 'opt/loss_bdv', 'opt/loss_up','elapsed_time',
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
        vert2 = coords.W.array.get()
    else:
        vert2 = coords.W.array
    ca_final = compute_curvature_sub(vert2,N,args.constrained_vert,verbose=True)
    print("\n\n (final,target) Curvature: ", [(round(ca_final[i].item(),5),args.target_curvature[j]) for i,j in enumerate(args.constrained_vert)])

    print("boundary squared-error: ", (np.sum( (fixed_coords-vert2[args.fixed_vert])**2 ) ))

    # output
    plydata['vertex']['x']=vert2[:,0]
    plydata['vertex']['y']=vert2[:,1]
    plydata['vertex']['z']=vert2[:,2]
    plydata.write(os.path.join(args.outdir,args.output))
    # graphs
    n = len(ca)
    sns.violinplot(x=np.array([0]*n ), y=[c.item() for c in ca_final])
    plt.savefig(os.path.join(args.outdir,"curvature_final.png"))
    plt.close()
    sns.violinplot(x=np.array([0]*n ), y=[c.item() for c in ca])
    plt.savefig(os.path.join(args.outdir,"curvature_init.png"))
    plt.close()
    error = [abs(ca_final[i].item()-args.target_curvature[j]) for i,j in enumerate(args.constrained_vert)]
    sns.violinplot(x=np.array([0]*n), y=error, cut=0)
    plt.savefig(os.path.join(args.outdir,"error_final.png"))
    plt.close()
    error = [abs(ca[i].item()-args.target_curvature[j]) for i,j in enumerate(args.constrained_vert)]
    sns.violinplot(x=np.array([0]*n), y=error, cut=0)
    plt.savefig(os.path.join(args.outdir,"error_init.png"))
    plt.close()

if __name__ == '__main__':
    main()
