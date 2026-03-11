# Tutorial: making mesh geometry differentiable


In the last two tutorials, we used `triangulax` to compute the gradients
of mesh-associated quantities like triangle or cell areas. However, you
may wonder: doesn’t the “discrete” nature of triangular meshes pose a
problem for differentiability? While the vertex (or face) positions
**v**<sub>*i*</sub> are smoothly varying, the mesh *connectivity* is
inherently discrete - two vertices *i*, *j* are either linked by an
edge, or they are not. And does this mean that we need to keep the mesh
connectivity fixed during simulations or optimizations?

### Related work

In response to the above concerns,
[several](https://arxiv.org/abs/2109.10695)
[authors](https://arxiv.org/abs/2404.13445v3) have proposed
“differentiable” meshes. In brief, the idea is to assign to all possible
mesh configurations over a set of *N* vertices a “probability”, which
varies smoothly as the vertex positions change. [For
example](https://arxiv.org/abs/2109.10695), one can compute a
probability weight for each possible face *i**j**k* from a score that
measures how close the face is to fulfilling the geometric criterion to
be included in the Delaunay triangulation of the vertex set. But this
approach has several downsides:

1.  Computing the probability weights can be costly/complicated, since
    the number of possible mesh configurations is very large.
2.  Treating surfaces embedded in 3D requires either mapping them to 2D
    or a tetrahedral mesh of the surface’s interior.
3.  It’s unclear how to compute arbitrary quantities (like Voronoi
    areas) from such a “probabilistic triangle soup”.

### Differentiability across topological modifications

In many cases, making the mesh probabilistic in this way is, however,
overkill. For concreteness, let’s consider some function
*f*({**v**<sub>*i*</sub>}|*G*) of the vertex positions
**v**<sub>*i*</sub> ∈ ℝ<sup>*d*</sup>, *i* = 1, ..., *N* and the mesh
connectivity graph *G* (for example, the mean triangle area). As long as
*G* is fixed, there is, of course, no issue, and *f* is a smooth
function of the **v**<sub>*i*</sub>.

Now suppose we want to dynamically adjust the mesh connectivity *G* as a
function of the vertex positions. For example, we can set *G* to be the
Delaunay triangulation of the vertex set ([a notion well-defined also in
3D](https://nmwsharp.com/media/papers/int-tri-course/int_tri_course.pdf)).
This means we need to “flip” an edge *i**j* whenever its Voronoi length
becomes negative (all mesh modifications that conserve the number of
vertices can be written as a sequence of flips).

From an abstract point of view, the space **R**<sup>*d* × *N*</sup> of
vertex configuration is now divided into “topological cells”, within
each of which the topology is constant. The boundaries of a cell
correspond to flips of an edge (so each cell is a node of the
triangulation’s [flip graph](https://en.wikipedia.org/wiki/Flip_graph)).
Our function *f* is smooth *within* each topological cell. We now need
to ask what happens if we cross a cell boundary. There are three
scenarios:

1.  *f* is smooth across cell boundaries. This is trivially the case for
    functions that don’t depend on *G*, but only on the vertex
    locations.

2.  *f* is continuous, but not smooth. A great example is the dual
    Voronoi length of a half-edge or the Voronoi area of a cell. The
    reason is that the dual length of an edge is 0 when the edge is
    flipped. Such *f* are thus continuous and piecewise smooth
    (“flip-continuous”), but may have “kinks” across topological
    modifications. The success of the `relu` activation function in
    machine learning suggests that this is nothing to worry about.

3.  *f* is discontinuous across cell boundaries. An example is the
    length of an edge, or the area of a triangle.

Below, we plot the three scenarios for a perfect triangular lattice in
2D undergoing shear, which eventually induces an edge flip.

#### Smoothed Delaunay

How can we mitigate scenario 3? Suppose, for example, we want to
optimize the energy
∑<sub>*i**j**k*</sub>(*a*<sub>*i**j**k*</sub> − *a*<sub>0</sub>)<sup>2</sup>
to make all mesh triangles equal-sized. The idea is that we do not have
to worry about *all* possible mesh configurations - only the ones of the
adjacent “topological cells”, which are precisely those that differ from
the current mesh *G* by a single edge flip.

Thus, for each half-edge *e*, we can compute a smooth probability weight
*p*<sub>*e*</sub> that measures how close it is to being flipped (for
example, you could take
*p*<sub>*e*</sub> = 1/(1 + *e*<sup>−ℓ<sub>*i**j*</sub><sup>*V*</sup>/*λ*</sup>),
where ℓ<sub>*i**j*</sub><sup>*V*</sup> is the dual Voronoi length, and
*λ* \> 0 a smoothing parameter). We can then sum over the possible
flipped configurations, weighted by *p*<sub>*e*</sub>. Luckily, most
quantities are *local* and only depend on a few neighboring mesh
elements, so that the number of flipped configurations to consider is
small (one may need to use something like softplus to convert the
different edge weights into probabilities). This procedure is easy to
implement using the half-edge mesh data structure and the gather-scatter
algorithm of
[`jax.numpy.ndarray.at`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.ndarray.at.html#jax.numpy.ndarray.at).

### “In the loop” mesh modifications

Using the above techniques - smoothed Delaunay or flip-continous
objective functions - we can get meaningful, continuous gradients across
topological modifications. Additionally, the array-update-based function
for the edge flips from the `topology` module is fully JAX compatible
(supports JIT-compilation) - thus, nothing prevents us from changing the
mesh topology within the “loop” of time/optimization steps.

#### What do the gradients *mean*?

<!-- WARNING: THIS FILE WAS AUTOGENERATED! DO NOT EDIT! -->

``` python
# to do: plot Voronoi area, edge length etc for sheared lattice
# implement smoothed Delaunay for area energy minimization with flips
```
