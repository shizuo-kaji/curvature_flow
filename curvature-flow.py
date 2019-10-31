#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import argparse,os

from plyfile import PlyData, PlyElement
import chainer
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer import training,datasets,iterators,Variable
from consts import optim,dtypes

## updater 
class Updater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.coords = kwargs.pop('models')
        params = kwargs.pop('params')
        super(Updater, self).__init__(*args, **kwargs)
        self.args = params['args']
        self.faces = params['faces']
        self.fixed_coords = self.coords.W.array[self.args.fixed_vert]  # it is copied!

    def update_core(self):
        opt = self.get_optimizer('opt')
        xp = self.coords.xp
        if self.args.optimise_cos:
            ca = compute_cos_angle(self.coords.W,self.faces,self.args.target_curvature,xp)
            loss = -sum([ca[i] for i in self.args.free_vert])/len(self.args.free_vert)
        else:
            K = compute_curvature(self.coords.W,self.faces,xp)
            loss = sum([(K[i]-self.args.target_curvature[i])**2 for i in self.args.free_vert])

        self.coords.cleargrads()
        loss.backward()
        opt.update(loss=loss)
        self.coords.W.array[self.args.fixed_vert] = self.fixed_coords

        chainer.report({'loss': loss.item()}, self.coords)

# Compute cosine of angle defect
def compute_cos_angle(vert,face,theta,xp):
    # TODO: it is better to iterate over vertex neighbours rather than over faces
    cc = xp.cos(theta)
    ss = xp.sin(theta)
    cos_angle=[Variable(cc[i:(i+1)]) for i in range(len(cc))]
    sin_angle=[Variable(ss[i:(i+1)]) for i in range(len(ss))]
    for f in face:
        n = len(f)
        id_p = xp.array([f[(i-1)%n] for i in range(n+1)])
        id = xp.array([f[i%n] for i in range(n+1)])
        id_n = xp.array([f[(i+1)%n] for i in range(n)])
        L = F.sum((vert[id_p] - vert[id])**2, axis=1)
        D = F.sum((vert[id_n] - vert[ id_p[:-1] ])**2, axis=1)
        c1 = (L[:n]+L[1:]-D)/(2*F.sqrt(L[:n]*L[1:]))
        s1 = F.sqrt(1-c1**2)
        for j,k in enumerate(f):
            c0, s0 = cos_angle[k],sin_angle[k]
            cos_angle[k] = c0*c1[j] - s0*s1[j]
            sin_angle[k] = c0*s1[j] + s0*c1[j]
    return cos_angle

# Compute gaussian curvature
def compute_curvature(vert,face,xp):
    K = [Variable(xp.array([2*xp.pi]))] * len(vert)
    for f in face:
        n = len(f)
        id_p = xp.array([f[(i-1)%n] for i in range(n+1)])
        id = xp.array([f[i%n] for i in range(n+1)])
        id_n = xp.array([f[(i+1)%n] for i in range(n)])
        L = F.sum((vert[id_p] - vert[id])**2, axis=1)
        D = F.sum((vert[id_n] - vert[ id_p[:-1] ])**2, axis=1)
        c1 = (L[:n]+L[1:]-D)/(2*F.sqrt(L[:n]*L[1:]))
        ac = F.arccos(c1)
        for j in range(n):
            K[f[j]] -= ac[j]
    return K

#####################################################################################
#-----------------------
def main():
    parser = argparse.ArgumentParser(description='Ranking learning')
    parser.add_argument('input', help='Path to ply file')
    parser.add_argument('--output', default="output.ply", help='Path to ply file')
    parser.add_argument('--target_curvature', '-K', default=None, type=str, help='target gaussian curvature')
    parser.add_argument('--fixed_vert', type=int, nargs="*", default=None,
                        help='indices of fixed vertices')
    parser.add_argument('--iteration', '-i', type=int, default=100,
                        help='Number of iterations')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--optimise_cos', '-cos', action='store_true', help='optimise cosine rather than curvature')
    parser.add_argument('--outdir', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--optimizer', '-op',choices=optim.keys(),default='Momentum',
                        help='optimizer')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--verbose', '-v', action='store_true',help='print debug information')
    args = parser.parse_args()

    chainer.config.autotune = True
    #chainer.print_runtime_info()

    # Read mesh data
    plydata = PlyData.read(args.input)
    vert = np.vstack([plydata['vertex']['x'],plydata['vertex']['y'],plydata['vertex']['z']]).astype(np.float64).T
    face = plydata['face']['vertex_indices']
    args.batchsize = 1 # dummy
    print(args)

    # set target curvature
    if args.target_curvature:
        args.target_curvature = np.loadtxt(args.target_curvature)
    else:
        args.target_curvature = np.full((len(vert),),4*np.pi/len(vert)) ## constant curvature with euler char = 2
    if len(args.target_curvature) != len(vert):
        print("Curvatures and vertices have different length!")
        exit(-1)

    # determine fixed vertices
    args.fixed_vert = np.where(args.target_curvature > 2*np.pi)[0]
    args.free_vert = list(set(range(len(vert))) - set(args.fixed_vert))
    print("\nvertices {}, faces {}, constrained vertices {}".format(len(vert),len(face),len(args.fixed_vert)))

    ######################################
    if args.verbose:
        ca = compute_curvature(vert,face,np)
        print("\n\n Initial Curvature: ", [round(ca[i].item(),5) for i in args.free_vert], "\n\n")

    coords = L.Parameter(vert)
    opt = optim[args.optimizer](args.learning_rate)
    opt.setup(coords)
    dummy_iter = chainer.iterators.SerialIterator(chainer.datasets.TupleDataset(np.arange(args.iteration)),args.batchsize)

    if args.gpu >= 0:
        coords.to_gpu() 

    updater = Updater(
        models=coords,
        iterator=dummy_iter,
        optimizer={'opt': opt},
        device=args.gpu,
        params={'args': args, 'faces': face}
        )
    trainer = training.Trainer(updater, (args.iteration//args.batchsize, 'iteration'), out=args.outdir)

    log_interval = 5, 'iteration'
    trainer.extend(extensions.LogReport(trigger=log_interval))
    if extensions.PlotReport.available():
        trainer.extend(extensions.PlotReport(['opt/loss'],'iteration', file_name='loss.png'))
    trainer.extend(extensions.PrintReport([
            'iteration', 'lr', 'opt/loss','elapsed_time',
        ]),trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=1))
    trainer.extend(extensions.observe_lr('opt'), trigger=log_interval)
    trainer.extend(extensions.LogReport(trigger=log_interval))
    ## annealing
    #        trainer.extend(extensions.ParameterStatistics(coords))
    #if args.optimizer in ['SGD','Momentum','CMomentum','AdaGrad','RMSprop','NesterovAG']:
    #    trainer.extend(extensions.ExponentialShift('lr', 0.5, optimizer=opt), trigger=(args.epoch/args.learning_rate_drop, 'epoch'))
    #elif args.optimizer in ['Adam','AdaBound','Eve']:
    #    trainer.extend(extensions.ExponentialShift("alpha", 0.5, optimizer=opt), trigger=(args.epoch/args.learning_rate_drop, 'epoch'))
    trainer.run()

    ####################################################
    ## result
    if args.gpu >= 0:
        vert2 = coords.W.array.get()
    else:
        vert2 = coords.W.array
    ca = compute_curvature(vert2,face,np)
    if args.verbose:
        print("\n\n final Curvature: ", [round(ca[i].item(),5) for i in args.free_vert])
        print("Target curvature:", args.target_curvature[args.target_curvature <= 2*np.pi])
#    print(vert2)
    # output
    plydata['vertex']['x']=vert2[:,0]
    plydata['vertex']['y']=vert2[:,1]
    plydata['vertex']['z']=vert2[:,2]
    plydata.write(args.output)

if __name__ == '__main__':
    main()
