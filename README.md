Mesh optimisation with a target Gaussian curvature
=============
This program deforms an input mesh (in the PLY format)
by modifying the coordinates of its vertices so that 
the resulting mesh has the target Gaussian curvature (defined as the angle defect)
specified at each point.

For this purpose, the discrete Ricci flow is often used.
The method is known to have preferable properties such as preserving the conformal structure
and convexity of the energy.
However, with the Ricci flow, it is difficult to specify constraints in terms of vertex coordinates.
This program takes a straightforward approach to optimise the vertex coordinates rather than the metric they define.

## Requirements
- Python 3: [Anaconda](https://www.anaconda.com/download/) is recommended
- Python libraries: Chainer, plyfile:  `pip install chainer plyfile`

# How to use
- For command-line options, look at
```
    python curvature-flow.py -h
```
- To deform the mesh in "dome.ply" and output "dome_deformed.ply" 
```
    python curvature-flow.py dome.ply -K dome_targetK.txt --output dome_deformed.ply -v -lr 1e-2 -e 100 -b 5
```
Note that depending on the input mesh, you have to specify an appropriate learning rate (-lr 1e-2)
and the number of iterations (-i 100).
Also, the number of vertices whose curvatures are optimised at a time (-b 5) may affect the convergence.
The default is set to be half the number of non-fixed vertices.

Gaussian curvature at each point is specified in the text file "dome_targetK.txt",
whose i-th row specifies the value of the curvature at the i-th vertex.
If the value is higher than 2pi, the value itself is ignored and the vertex is considered to be fixed
(that is, its coordinates remain unchanged after deformation).
The value -99 has a special meaning; the vertex is not constrained and can have an arbitrary curvature.


# LIMITATION
- In the output PLY file, only vertex coordinates are modified from the original PLY.
So other properties such as normal vectors are intact, which may result in a wrong rendering.
- Currently, the program is very slow.
