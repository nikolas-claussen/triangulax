# triangulax pre-release code review

## Still open

Everything not listed here has been fixed on branch `pre-release-fixes` (14 commits);
fixed items are annotated inline with their commit.

- **B16.** Hard-coded absolute `1e-12` zero-division guards break small-scale meshes
  *(deferred by request: to be handled by a separate zero-division-guarding overhaul.
  Note this is also the sole remaining cause of B11's residual small-scale error.)*
- **B25.** Medium-severity items — 22 of 26 fixed, 4 still open (see the table)
- **E.** Omissions — the "easy" items are now implemented (E2, E3, E4, E5, E6, E8).
  Still open by request: **E9** (padded one-ring traversal — declined: gather/scatter
  is the better pattern), **E10** (spatial acceleration for closest-point queries),
  and a periodic FE gradient.

Status: all 12 source notebooks pass `nbdev_test`; `ruff` is clean; the package builds
with all 14 modules. `__version__` is still `0.0.2` and the stale `dist/` artifacts are
still present — release tasks you said you would handle.

---

## A. Release blockers

### A1. Three public functions are broken on arrival for any `pip install` user [V]

> **[FIX IMPLEMENTED IN COMMIT 1a2802a]** converted the three igl call sites to `np.asarray(..., dtype=np.int64)`; verified under JAX's default (x64-off) config, and covered by a new subprocess smoke test.
`mesh.py:374, 332, 415`

`test_mesh_validity`, `HeMesh.bdry_loops` and `connect_boundary_to_infinity` pass JAX
arrays straight to igl, whose nanobind signatures require `int64`. Every notebook sets
`jax_enable_x64=True` in a hidden cell, so this is invisible in development. With JAX's
default config:

```
jax_enable_x64 = False
  HeMesh.from_triangles            OK
  hemesh.faces                     OK
  hemesh.is_bdry_he                OK
  iterate_around_vertex            OK
  test_mesh_validity               ** TypeError: is_edge_manifold(): incompatible function arguments
  hemesh.bdry_loops                ** TypeError: boundary_loop_all(): incompatible function arguments
  connect_boundary_to_infinity     ** TypeError: boundary_loop_all(): incompatible function arguments
```

Fix: convert at each igl boundary, as `get_half_edge_arrays_vectorized:64` already does —
`np.asarray(self.faces, dtype=np.int64)`. Three one-line changes.

### A2. `requires-python = ">=3.9"` but the code needs 3.10+ [V]

> **[FIX IMPLEMENTED IN COMMIT 40f4084]** `requires-python = ">=3.10"`; also dropped the unused `lineax` dependency and replaced the nbdev boilerplate PyPI keywords/classifiers.
`pyproject.toml:10`

`mesh.py` and `triangular.py` use PEP 604 unions in eagerly-evaluated annotations
(`str | Path`, `int|None`, and the dataclass field `tuple[()] | tuple[int]`) with no
`from __future__ import annotations`. Confirmed they are evaluated at import:

```
read_obj filename annot:   str | pathlib.Path
HeMesh.inf_vertices annot: tuple[()] | tuple[int]
```

pip installs happily on 3.9, then `import triangulax.mesh` raises `TypeError`.
Fix: `requires-python = ">=3.10"` (matches your own copilot-instructions).

### A3. `compute_periodic_cotan_laplace` raises on every call [V]

> **[FIX IMPLEMENTED IN COMMIT 24a33ac]** removed the spurious leading underscore; also renamed `distance_function` -> `displacement_fn`. Now verified against the non-periodic operator and a Fourier mode.
`linops.py:173`

Calls `per._get_periodic_cotan_weights_per_edge`; the actual name has no underscore.

```
periodic has '_get_periodic_cotan_weights_per_edge': False
periodic has  'get_periodic_cotan_weights_per_edge': True
CALL -> AttributeError: module 'triangulax.periodic' has no attribute '_get_periodic_cotan_weights_per_edge'
```

100% non-functional. Survived because notebook 06 contains **zero asserts** and never
calls it. Fix: delete the leading underscore. (The rest of the function is correct — with
the name patched it reduces to `compute_cotan_laplace` to 3.6e-15.)

### A4. 4 of 12 source notebooks fail `nbdev_test` [V]

> **[FIX IMPLEMENTED IN COMMIT 1a2802a, 2c5fc92, 7c9312a, df51fc4]** all four causes fixed (relative mesh paths, `#| notest` cells leaving later cells with undefined names, and two jaxtyping annotations too narrow for the notebook's own calls). All 12 source notebooks now pass, run twice in parallel to confirm stability.

```
nbdev Tests Failed On The Following Notebooks:
  01_triangular_meshes.ipynb      cell 30: jaxtyping TypeCheckError in generate_poisson_points
                                  (called as generate_poisson_points(n, *L) with L a jax array,
                                   but limit_x/limit_y are annotated `float`)
  02_halfedge_datastructure.ipynb cell 63/67: FileNotFoundError 'test_meshes/disk.obj'
                                  (every other cell uses '../test_meshes/...')
  09_algorithms.ipynb             NameError: 'noisy_vertices' is not defined
  10_simulation.ipynb             cell 26: NameError: 'tempfile' is not defined
```

Note `nbdev_test` exits 0 despite these, so they will not fail CI as configured.
Also: running the suite rewrites the tracked `nbs/test_meshes/disk_write_test.obj`.

---

## B. Correctness bugs

Ordered by severity. Two of these share a single root cause (B1 → B2).

### B1. `get_dihedral_angles` does not mask boundary half-edges [V]  — HIGH

> **[FIX IMPLEMENTED IN COMMIT 27ee56b]** boundary edges now return 0 via `jnp.where(hemesh.is_bdry_edge, ...)`; closed meshes bit-identical.
`geometry.py:114`

`normals[hemesh.heface]` with `heface == -1` silently indexes the **last face**. On a
curved open mesh (disk lifted onto a unit sphere):

```
max|theta| on boundary half-edges : 0.916   (should be 0/undefined)
max|theta| on interior half-edges : 0.197
H interior mean = 1.0058 (true 1.0);  H boundary mean = -4.2538
```

`get_dihedral_bending_energy` is safe (it masks `~is_bdry_edge`); `get_mean_curvature_dihedral`
and `get_helfrich_energy` are not.

### B2. `get_helfrich_energy` is 11x wrong on any open mesh, with non-local errors [V] — CRITICAL

> **[FIX IMPLEMENTED IN COMMIT 27ee56b]** fixed by B1 with no change to elastic.py. E_helfrich on the bent disk: 0.9177 -> 0.0917 (interior truth 0.0821); the non-local dependence on the last face is gone.
`elastic.py:306` — consequence of B1

Disk bent onto a cylinder of R=2 (exact H=0.25, correct E≈0.082):

```
E_helfrich (library)       = 0.9177
interior-only contribution = 0.0821   <- physically correct
boundary share of total    = 91.0%    (11.2x too large)
H boundary mean = -1.4504

non-locality: perturb vertex 63 (a corner of the LAST face)
  E: 0.9177 -> 12.9280
  H at an untouched boundary vertex: -1.7537 -> -10.5612
```

So forces are wrong too, and the mesh acquires a spurious long-range coupling to whichever
triangle happens to be last in `faces`.

**One-line fix resolves both B1 and B2** (verified):
```python
# geometry.py get_dihedral_angles
return jnp.where(hemesh.is_bdry_edge, 0.0, jnp.arctan2(sin_theta, cos_theta))
```
```
AFTER the fix:
  E_helfrich = 0.0917  (was 0.9177; interior truth 0.0821)
  H boundary mean = 0.1411 (was -1.4504)
  non-locality: H at untouched bdry vertex 0.2567 -> 0.2567 (was -1.75 -> -10.56)
  closed sphere unchanged: True    <- no regression on closed meshes
```

### B3. Batch edge flips silently produce non-manifold meshes [V] — CRITICAL

> **[FIX IMPLEMENTED IN COMMIT 8e55e46]** each flip is re-checked with `can_flip_edge` against the scan CARRY. Stress test: 8/12 trials corrupted -> 0/12.
`topology.py:108` (`flip_all`), `:94` (`flip_by_id`), `:143` (`flip_n_shortest`)

`to_flip` is evaluated once against the *original* mesh but flips are applied sequentially
under `lax.scan`. An earlier flip can invalidate a later one; `flip_edge` applies it anyway.

```
random simultaneous-flip stress (60 interior edges, 12 trials, disk.obj):
   current flip_all corrupted : 8/12
   with can_flip_edge recheck : 0/12
```

**Fix** (verified — re-check against the carry `hh`, not the closed-over `hemesh`):
```python
def scan_fun(hh, e):
    do = to_flip[e] & hemesh.is_unique[e] & can_flip_edge(hh, e)
    return jax.lax.cond(do, lambda x: flip_edge(x, e), lambda x: x, hh), None
```

### B4. `fix_delaunay` corrupts meshes at moderate noise [V] — CRITICAL

> **[FIX IMPLEMENTED IN COMMIT 8e55e46]** follows from B3; `fix_delaunay` now stays valid at every noise level tested, and the test is no longer `#| notest` and asserts mesh validity.
`algorithms.py:190` — consequence of B3

```
disk.obj    noise=0.025  flips=15   validity=OK
            noise=0.06   flips=68   ** non-edge-manifold, 2 duplicate directed hes
            noise=0.10   flips=143  ** non-edge-manifold, 23 duplicate hes, nxt[prv] broken
```
The notebook's own test sits at noise 0.025 — just below the threshold. With the B3 fix
applied, all cases return clean. Your cached refinement meshes were checked and are fine.

### B5. Flipping a boundary edge corrupts the mesh; `check_boundary=True` is dead code [V] — CRITICAL

> **[FIX IMPLEMENTED IN COMMIT 8e55e46]** removed the dead `check_boundary` parameter from `flip_edge`/`collapse_edge`; the `can_*_edge` re-check now screens boundary edges for real.
`topology.py:43-44`, `:56-57` (same at `collapse_edge:210`)

`face_incident.at[jnp.array([heface[e], heface[twin]])]` — with `heface == -1`, JAX wraps
and overwrites the **last face**.

```
face_incident[-1] before: 199  after: 357
validity after boundary flip: ['orig!=dest[twin]', 'nxt[prv]']
check_boundary=True -> TracerBoolConversionError
```
The guard can never fire: `flip_edge` is `@jax.jit`, so a Python `assert` on a traced value
raises. Fix: delete the `check_boundary` parameter from both functions and rely on the
`can_*_edge` re-check from B3 (which does reject boundary edges).

### B6. Parallel transport returns the wrong angle entirely [V] — CRITICAL

> **[FIX IMPLEMENTED IN COMMIT 27ee56b]** switched both transport functions to `get_signed_angle_between_vectors`. Antisymmetry now holds to 2e-16 and holonomy = -(angle defect) mod 2pi. The two test cells that asserted the bug were replaced.
`geometry.py:503`, `:536`

Both transport functions use the **unsigned** `get_angle_between_vectors` (range `[0,π]`)
and additionally negate the twin coordinates. Result is `π − |correct signed angle|`.

```
max|r - r[twin]| = 0.0       <- symmetric (WRONG)
max|r + r[twin]| = 4.3435    <- should be 0 (transport must be antisymmetric)
```
Note `RᵀR = I` and `det R = 1` hold for *any* angle, so those are vacuous tests — only
antisymmetry and holonomy discriminate.

**Fix** (verified — the right primitive already exists):
```python
transport_angle = jax.vmap(trig.get_signed_angle_between_vectors)(edge_vec_face, edge_vec_face_twin)
```
```
AFTER: antisymmetry max|r + r[twin]| = 1.19e-07  (float32 eps)
       max|R @ coords_f - coords_g| = 1.19e-07   (genuinely maps frame f -> frame g)
       one-ring holonomy = -(angle defect), mod 2pi  (e.g. 6.009346 - 2pi = -0.27384 = -defect)
```
The notebook's test cells 56/58 assert `transports - transports[twin] == 0` — i.e. they
assert the bug. Replace with the antisymmetry + holonomy checks.

### B7. 2D triangle/vertex normals are silently wrong [V] — HIGH

> **[FIX IMPLEMENTED IN COMMIT 27ee56b]** `get_triangle_normals`/`get_vertex_normals` documented 3d-only with shape asserts; added `get_triangle_orientations` returning +1/-1 per face for 2d meshes.
`geometry.py:66`, `:74`

In 2D `get_oriented_triangle_areas` returns shape `(n_faces,)`, so
`jnp.linalg.norm(..., axis=-1)` reduces the only axis to a **single global scalar**, and
every normal is divided by it.

```
disk.obj (2D): get_triangle_normals -> [0.029, 0.091, 0.051, 0.042, 0.055, ...]
docstring promises "In 2d, this just returns +/-1";  max|n| = 0.1028
get_oriented_triangle_areas shape in 2D: (224,)   (type hint says "n_faces dim")
```
Matters because 2D is the stated primary use case, and `algorithms.py:232` calls
`get_vertex_normals`. 3D is exact (matches igl to 1e-16). Related: `get_edge_normals` on
2D input silently returns an `(n_hes, n_hes)` matrix instead of erroring.

### B8. `test_mesh_validity` certifies corrupt meshes [V] — HIGH

> **[FIX IMPLEMENTED IN COMMIT 1a2802a]** `jnp.allclose` -> `jnp.array_equal` in `test_mesh_validity` and `HeMesh.__eq__`; also added the unmasked `nxt[prv]`/`prv[nxt]` checks, so a scrambled boundary loop is now rejected.
`mesh.py:361, 367-373`

All eight assertions (and `HeMesh.__eq__`) compare **integer index** arrays with
`jnp.allclose`, whose `rtol=1e-5` makes off-by-one errors invisible above ~1e5.
`torus_high_resolution.obj` has 221 184 half-edges.

```
twin array actually differs: True
jnp.allclose says equal    : True    <- off-by-one hidden
HeMesh.__eq__ (should be False): True
twin[twin]==arange broken  : True
test_mesh_validity accepts it: True
```
Fix: `jnp.array_equal` at all 9 sites. Also add `nxt[prv] == arange` and
`prv[nxt] == arange` — currently the boundary loop linkage is never validated at all
(a scrambled boundary cycle passes, and then `iterate_around_vertex` reports a
valence-4 vertex as having a 56-half-edge fan).

### B9. `bdry_loops` returns opposite winding under the two boundary conventions [V] — HIGH

> **[FIX IMPLEMENTED IN COMMIT 1a2802a]** reversed the inf-vertex branch so both conventions match the face orientation; asserted in the notebook.
`mesh.py:330-333`

```
disk.obj signed area, heface==-1 convention:  3.1176
         signed area, inf-vertex convention: -3.1176
```
The `-1` branch (via igl) is the correct one. Fix: reverse the inf branch.

### B10. `mass_matrix_sparse` uses signed exact Voronoi areas, not mixed [R] — HIGH

> **[FIX IMPLEMENTED IN COMMIT 24a33ac]** default `area_type="voronoi"` is now Meyer's mixed area (what igl computes). Agreement with igl: 2.8e-3 -> ~1e-17. Old behaviour available as `"voronoi_exact"`; unknown values now raise ValueError.
`linops.py:206-210, 231-235`

Disagrees with `igl.massmatrix(VORONOI)` by 2.2% (disk), 5.2% (sphere_fine_poor), 1.2%
(torus); igl's VORONOI is Meyer's *mixed* area = your `get_voronoi_areas_robust`. On an
obtuse mesh the areas go negative, `mass_matrix_inv` goes negative, and `(M - dt·L)` loses
positive-definiteness — which invalidates the notebook's own implicit-diffusion example
that tags the operator `lineax.positive_semidefinite_tag`.

### B11. `get_closest_point_on_triangle` silently wrong for small-scale meshes [R] — HIGH

> **[FIX IMPLEMENTED IN COMMIT abae5d2]** normalized the face normal in `get_closest_point_on_triangle` and `find_closest_faces` with a scale-free double-where, and made the degeneracy tests scale-free. **Partial:** the residual error below scale ~1e-3 is entirely due to `trigonometry.get_barycentric_coordinates` clipping its length^4 denominator at 1e-12 — verified by substituting a relative guard there, which makes the queries exact (7e-17) down to scale 1e-5. That guard is B16 and is deferred.
`interp.py:63-65`

`normal = cross(b-a, c-a)` is unnormalized and `project_out_vector` clips `|n|²` at 1e-12,
so once triangle area ≲ 5e-7 the projection collapses. End-to-end vs
`igl.point_mesh_squared_distance` on sphere.obj: scale 1 → 4.4e-16; scale 1e-3 → **2.7x
the mesh radius**. Relevant for tissue meshes in microns/metres. Fix: normalize the normal.

### B12. FE gradient produces NaN gradients on degenerate faces [R] — HIGH

> **[FIX IMPLEMENTED IN COMMIT abae5d2]** made the denominators safe before dividing in `_fe_grad_phi_2d/3d` (jnp.where masks the primal but reverse mode still differentiates the untaken x/0 branch). 6-9 NaNs -> 0 on a mesh with a collapsed triangle.
`linops.py:250-252, 270-272`

Classic single-`where`: the primal is masked but the VJP of the untaken branch
differentiates `x/0`. 6–9 NaNs on a collapsed triangle. Matters because triangles do
collapse mid-optimization and one NaN poisons the whole gradient. Fix: safe-denominator
double-`where` (the pattern `trig.get_triangle_area_from_sides` already uses correctly).

### B13. `compute_divergence_2d/3d` crash on the vector fields `compute_gradient_*` produces [V] — HIGH

> **[FIX IMPLEMENTED IN COMMIT 24a33ac]** area weighting reshaped to broadcast against any trailing axes; scalar results unchanged.
`linops.py:409, 430`

`areas[:, None]` only broadcasts against rank-2 `contrib`:
```
grad shape: (224, 2, 2)
div -> ValueError: Incompatible shapes for broadcasting: shapes=[(224, 3, 2), (224, 1)]
```
The docstring advertises exactly this composition. Fix:
`areas.reshape((-1,) + (1,) * (contrib.ndim - 1))`.

### B14. `get_polygon_area` sign is inverted vs its docstring [V] — HIGH

> **[FIX IMPLEMENTED IN COMMIT 483bd3c]** corrected the shoelace roll direction; the compensating minus sign in `geometry._get_cell_areas_traversal` was removed in the companion commit 27ee56b.
`trigonometry.py:99, 111-112`

```
CCW unit square -> -1.0   (docstring: "Positive for counter-clockwise")
CCW triangle    -> -0.5
get_oriented_triangle_area on the same triangle -> +0.5   (disagrees)
```
**Coupled change:** `geometry.py:286` compensates with an explicit `-`. Fix both together
or `_get_cell_areas_traversal` flips sign.

### B15. `get_voronoi_corner_area` destroys the sign on obtuse triangles [V] — HIGH

> **[FIX IMPLEMENTED IN COMMIT 483bd3c]** function removed. It was not called anywhere in the library.
`trigonometry.py:124-126`

`jnp.linalg.norm` takes `|signed dual area|`; corner areas then don't sum to the triangle area:
```
corner areas: [11.35, 7.52, 3.72]  sum = 22.60
true triangle area = 0.10
```
Two defects: the midpoints are `(a-c)/2` instead of `(a+c)/2` (currently masked by the
`norm`), and the `norm` itself. Fix both together.

**Scoping correction:** this function is **not called anywhere in the library** —
`geometry.get_voronoi_corner_areas` (plural) is a separate, correct cotangent
implementation that matches `igl.massmatrix(VORONOI)` to 1e-17. So this is a broken
exported primitive, not a corrupted mass matrix. [V]

### B16. Hard-coded absolute `1e-12` guards break small-scale meshes [R] — HIGH
`trigonometry.py:38, 145, 164, 180, 233, 298, 325, 454`

The clips compare dimensional quantities (16·Area², |b|², 4·Area) against an absolute
constant, so correctness depends on coordinate units. A well-shaped equilateral triangle
with side 1e-4 is treated as degenerate:
```
scale 1e-3  circumcenter rel.err 9.4e-17
scale 1e-4  circumcenter rel.err 9.997e-01   <- garbage, silent
```
`get_barycentric_coordinates` is already wrong at scale 1e-3. Fix: make the guards
relative, or at minimum document the O(1)-coordinates assumption.

### B17. `get_periodic_delaunay_faces` silently returns invalid meshes for coarse point sets [V] — HIGH

> **[FIX IMPLEMENTED IN COMMIT 7c9312a]** now asserts `n_faces == 2*n_vertices` (Euler characteristic 0) and fails loudly instead of silently returning a holed mesh; documented under Raises. Silently-invalid results: 0/20 at every size tested.
`triangular.py:440-441`

`np.unique(np.sort(faces, axis=1), axis=0)` dedups by vertex-id *triple*, but on a torus
two genuinely distinct triangles can share one. Valid ⇔ `n_faces == 2n` and χ = 0:
```
n     invalid/20   (L = [1.7, 1.1])
  8    20/20
 12    19/20
 16    16/20
 24     4/20
 32     0/20
 64     0/20
```
The notebook tests only n ∈ {32, 64, 96} — just above the threshold. Since the
global-index face list genuinely cannot express this, the right fix is to **fail loudly**:
`assert faces.shape[0] == 2 * n_vertices`.

### B18. `kabsch_align` is hardcoded to 3D [R] — HIGH

> **[FIX IMPLEMENTED IN COMMIT 2c5fc92]** reflection correction now works in any dimension; verified det(R) = +1 in 2d and 3d including against a mirrored target.
`algorithms.py:58` — `R = (U * jnp.array([1.0, 1.0, d])) @ Vt` raises on 2D input, though
the signature and docstring both promise `dim`. Fix:
`R = (U * jnp.ones(U.shape[-1]).at[-1].set(d)) @ Vt`.

### B19. `smooth_vertices_laplacian` indexes `areas[-1]` for boundary half-edges [R] — HIGH

> **[FIX IMPLEMENTED IN COMMIT 2c5fc92]** boundary half-edges masked to zero weight (72 of 708 half-edges on disk.obj were wrong).
`algorithms.py:224` — same `heface == -1` wrap. Boundary vertices land up to 0.033
(`bc='free'`) / 0.205 (`bc='slide'`) off, on a mesh with mean edge length 0.183. Masked
entirely by the default `bc='fixed'`, which is why the notebook never caught it.

### B20. `can_flip_edge` / `can_collapse_edge` miss boundaries on inf-vertex meshes [R] — HIGH

> **[FIX IMPLEMENTED IN COMMIT 8e55e46]** both predicates now use `hemesh.is_bdry_edge`, which handles both boundary conventions.
`topology.py:81, 185` — both test `heface != -1`, but no half-edge has `heface == -1` under
the inf convention, so both approve genuine boundary edges. Fix: use `hemesh.is_bdry_edge`.

### B21. `collapse_edge` is unusable on inf-vertex meshes [R] — HIGH

> **[FIX IMPLEMENTED IN COMMIT 8e55e46]** works now that `@jax.jit` is removed module-wide; documented that it cannot be jitted on an inf-vertex mesh.
`topology.py:263` — `int(remap_v[v])` on a traced value under `@jax.jit` →
`ConcretizationTypeError`. Recommend an explicit `NotImplementedError` when
`hemesh.has_inf_vertex`.

### B22. `can_collapse_edge` is O(n_hes²) [R] — HIGH

> **[FIX IMPLEMENTED IN COMMIT 8e55e46]** replaced the n_hes x n_hes comparison with a per-vertex scatter: 9.74 ms -> 0.058 ms on sphere_finer.obj.
`topology.py:191` — `nbrs_v0[:, None] == nbrs_v1[None, :]`. 0.39 ms → 9.74 ms going from
3 456 to 15 360 half-edges; extrapolates to ~2 s per single check at 221k. A scatter-based
version is 0.03–0.04 ms and agrees on every half-edge tested.

### B23. `flip_all` scans all `n_hes` — this is your known ~47 s problem [R] — HIGH

> **[FIX IMPLEMENTED IN COMMIT 8e55e46]** added an optional `max_flips` that scans only the candidates: 30.9 s -> 2.05 s (15x) on torus_high_resolution.obj, identical result. **JIT impact:** the default (`max_flips=None`) keeps the old behaviour and is jittable with no static arguments; passing `max_flips` sets the scan length and hence array shapes, so it must be static (`jax.jit(flip_all, static_argnames=['max_flips'])`).
`topology.py:110` — cost is `scan_length × n_hes`. Scanning only the candidate ids (with a
static `max_flips`) gives 21 405 ms → 3 104 ms on `torus_high_resolution.obj`, and folds in
the B3 correctness fix for free. Signature change; if unacceptable, the one-line
`can_flip_edge` re-check alone restores correctness at current cost.

### B24. `.obj` infinity-vertex round trip is broken two ways [R] — HIGH

> **[FIX IMPLEMENTED IN COMMIT abae5d2]** sentinel changed 1e300 -> 1e30 (representable in float32; the old one overflowed to literal "inf" with x64 off, which igl could not read back), the write path forces float64, and the reader now compares against half the sentinel instead of `>` it. Round trip verified under both precisions.
`triangular.py:203-204, 243-244` — (a) `> _INF_SENTINEL` is strict, and the writer emits
exactly `1e+300`, so `1e300 > 1e300` is False; (b) with x64 off (the default!) the
float32 cast overflows and writes literal `inf`, producing a file igl cannot read back at
all. Fix: `>=` comparison, force float64 in the write path, or use a `1e30` sentinel.

### B25. Medium-severity items (abbreviated)

> **[MOSTLY FIXED]** 22 of the 26 rows are now fixed. Beyond the 11 noted previously, commit `abae5d2` fixed: `linops.py:450` (output-shape guard), `elastic.py:334,347` (tangential basis layout), `elastic.py:203` (0/0 NaN), `elastic.py:110,65` (sqrt of a negative det), `geometry.py:440` (zero guard), `geometry.py:443 vs 471` (handedness), `geometry.py:154` (boundary mask), `geometry.py:123` (closed-mesh check), `mesh.py:319` (infinite loop), `mesh.py:57-134` (silent corruption), `mesh.py:297` (dtype), `mesh.py:347,553` (numpy on load), `mesh.py:356` (unhashable), `periodic.py:36` and `periodic.py:201` (docs). `mesh.py:572` (`GeomMesh.__eq__` raising) was fixed by the rewrite in `d6a4131`. **Still open:** the four rows whose fix is a zero-division guard, i.e. B16 territory.

> **[PARTIALLY FIXED]** 11 of the 26 rows below are fixed: `linops.py:84` and `linops.py:206` (24a33ac); `mesh.py:378` (1a2802a); `topology.py:65,314`, `topology.py:144` and the leaked tracer in `collapse_edge`'s `MeshReindexMap.info` (8e55e46); `algorithms.py:241` and `algorithms.py:235` (2c5fc92); `simulation.py:120` and the two `simulation.py` docstring errors (df51fc4); `triangular.py:198` and `triangular.py:455` (7c9312a). The remaining rows are still open.

| Location | Issue | Verified |
|---|---|---|
| `linops.py:84` | `jnp.abs(k)` makes `diag_jsparse` + both mass matrices un-jittable (`ConcretizationTypeError`); fix is `abs(k)` | [V] |
| `linops.py:450-483` | `linear_op_to_sparse` never checks the op's actual output shape → BCOO with out-of-bounds indices, silently dropped rows | [R] |
| `linops.py:206` | invalid `area_type` → `UnboundLocalError`; unguarded `1/cell_areas` | [R] |
| `elastic.py:334, 347` | `make_tangential_energy`/`vertices_from_tangential` reject the basis their own docstring recommends: `get_vertex_tangent_basis` returns `(2, n, 3)`, the einsum needs `(n, 2, 3)` | [V] |
| `elastic.py:203` | `get_dihedral_bending_energy` → NaN when both adjacent triangles collapse (0/0) | [R] |
| `elastic.py:110, 65` | `sqrt(det)` returns NaN instead of the intended `+inf` barrier when `det` rounds negative — destroys the neo-Hookean inversion barrier exactly where an optimizer probes | [R] |
| `geometry.py:440` | `get_face_tangent_basis` is the only normalization in the module without a zero-guard → NaN *values* (not just gradients) | [R] |
| `geometry.py:443 vs 471` | face and vertex tangent bases have **opposite handedness** (+1 vs −1); the face docstring contradicts its own code | [R] |
| `geometry.py:154` | `get_dual_he_length` doesn't mask boundary (its sibling `get_oriented_dual_he_length` does) | [R] |
| `geometry.py:123` | `get_volume` silently returns an origin-dependent number on an open mesh (0.58 → 50.89 under a +100z translation) | [R] |
| `mesh.py:319` | `iterate_around_vertex` loops forever on a vertex with no outgoing half-edge (`incident == -1`) | [R] |
| `mesh.py:57-134` | `from_triangles` silently builds corrupt meshes from non-manifold / duplicate-face / unreferenced-vertex input; cryptic `IndexError` on degenerate/inconsistently-oriented input | [R] |
| `mesh.py:297` | empty `inf_vertices` → `jnp.array(())` is float; every boundary predicate dies under `jax_numpy_dtype_promotion=strict` | [R] |
| `mesh.py:347, 553` | `HeMesh.load`/`GeomMesh.load` return **numpy** arrays → `.at[]` fails downstream; `GeomMesh.load` uses `jnp.load` on an `.npz`, which does nothing | [R] |
| `mesh.py:378` | `_canonical_faces_np` overflows int32 above ~1300 vertices → 73 key collisions on `torus_high_resolution`, weakening the topology round-trip tests | [R] |
| `mesh.py:356` | `HeMesh` is unhashable (defining `__eq__` sets `__hash__ = None`), blocking `static_argnums` | [R] |
| `mesh.py:572` | `GeomMesh.__eq__` **raises** on a shape mismatch instead of returning False | [R] |
| `periodic.py:36` | minimum-image silently fails for edges longer than `L/2`, undetected and undocumented (real for coarse or strongly sheared meshes) | [R] |
| `periodic.py:201` | `get_periodic_voronoi_face_positions` returns unwrapped positions without saying so (its sibling documents it) | [R] |
| `topology.py:65, 314` | int64→int32 scatter `FutureWarning`: *"In future JAX releases this will result in an error"* — triggered by every scan path | [R] |
| `topology.py:144` | `flip_n_shortest` over-reports `did_flip` under length ties (8 reported vs 5 actual) — tutorial 03 keys its flip cooldown off this | [R] |
| `algorithms.py:241` | `bc='slide'` → all-NaN gradients (0/0 in the tangent normalization) | [R] |
| `algorithms.py:235` | `bc` unvalidated: `bc='Fixed'` silently means `'free'` | [R] |
| `simulation.py:120` | `chunked_simulate` crashes on `n_steps=0` / `chunk_size<=0` (`simulate` handles them) | [R] |
| `triangular.py:198` | `TriMesh.read_obj` hardcodes `float64`/`int64` → 2 `UserWarning`s + silent downcast on every call | [R] |
| `triangular.py:455` | `jnp.array(faces, dtype=jnp.int64)` → truncation warning on every `get_periodic_delaunay_faces` call | [V] |

---

## C. Systemic issue: the notebook tests mostly don't assert

> **[FIX IMPLEMENTED IN COMMIT 3a8f841]** Display cells converted to real assertions across notebooks 01, 02, 04, 05, 06 and 09. `06_linear_operators` went from 0 asserts to 30. Added the missing curved-mesh-with-boundary fixture (verified to fail against the pre-fix code), an x64-off subprocess smoke test, the first tests for the periodic Laplacian and `diag_jsparse`, and mesh-validity assertions after edge flips. Two cells that asserted the *bug* (parallel transport) were replaced.

This is the single highest-leverage fix in the whole review, and it is the direct cause of
A3, B3, B5 and B6 shipping.

`nbdev_test` only fails on **exceptions**. A cell ending in `print("Equal?", False)` or a
bare `jnp.allclose(...)` expression passes silently.

| notebook | asserts |
|---|---|
| `06_linear_operators.ipynb` | **0 asserts, 15 prints** |
| `02_halfedge_datastructure.ipynb` | 3 assert-cells / 61 code cells |
| `04_adjacency_operators.ipynb` | 2 assert-cells / 25 code cells |
| `00_trigonometry.ipynb` | cells 15–20 and 28–30 assert nothing |
| `05b` (periodic) | 37 asserts — the good example |

Notebook 06's committed output literally shows `mass matrix rel. error: 0.0217` printed
directly beneath a line reading `1.87e-16` (finding B10). Converting those 15 prints to
asserts would have caught A3 and B10 before release.

Tests that assert the *wrong* thing:
- `05_geometric_quantities` cells 56/58 assert `transports - transports[twin] == 0` —
  that is the symptom of B6. Should be `+`.
- `08_elasticity` cell 20 asserts `E(flat disk) == 0` for the dihedral bending energy —
  but a flat mesh has zero dihedral angles *everywhere*, including on the fake boundary
  half-edges, so it passes whether or not the mask exists.

Structural blind spots:
- **No curved mesh with a boundary anywhere.** All 3D tests use closed meshes; all boundary
  tests use the planar disk. A single sphere-cap fixture catches B1 and B2 immediately.
- **No test runs with x64 off** — the config every PyPI user gets (A1).
- **No AD or degeneracy tests.** No cell calls `jax.grad` in `00_trigonometry` or
  `06_linear_operators`, in a library whose stated purpose is gradient-based optimization.
- **No mesh-validity assertion after any edge flip** (B3, B5). `assert msh.test_mesh_validity(...)`
  in cells 16 and 26 of notebook 03 is the cheapest possible guard.
- **No negative tests** — nothing asserts that `test_mesh_validity` *rejects* a corrupt mesh.
- **Untested public functions**, non-exhaustively: `kabsch_align` (no test cell at all),
  `can_flip_edge`, `can_collapse_edge`, `can_split_vertex`, `flip_by_id`, `flip_n_shortest`,
  `get_volume`, `get_dihedral_angles`, `get_triangle_normals`, `get_vertex_normals`,
  `get_edge_normals`, `compute_per_face_jacobian`, `get_rot_mat`, `get_perp_2d`,
  `quaternion_to_rot_mat`, `get_tetrahedron_volume`, `diag_jsparse`, and the four
  normal/tangential energy helpers.
- `fix_delaunay`'s test is marked `#| notest`, so it never runs.

---

## D. Style & quality

> **[FIX IMPLEMENTED IN COMMITS 483bd3c..40f4084]** `ruff` is clean (23 findings -> 0): all unused imports removed (test-only ones moved to non-exported cells), the `triangular.py` Bool annotation corrected, `op: callable` -> `Callable`, and the two `E741`/`E731` findings resolved. Contradictory docstrings fixed in geometry, algorithms, simulation and linops. Packaging metadata fixed in 40f4084. **Still open:** `__version__` is still `0.0.2` (the version number is your call — the removal of `get_voronoi_corner_area` and `check_boundary` is API-breaking, so 0.1.0 may fit better than 0.0.3); the stale `dist/triangulax-0.0.2.*` artifacts from April are still present and would trip `twine upload dist/*`; the tracked `Screenshot 2026-02-10….png` at repo root; and the README quickstart still reads `test_meshes/disk.obj`, which a pip user does not have.

- **`ruff`: 23 errors.** 20 unused imports (`topology.py` alone has 7, including an absolute
  `import triangulax.mesh as msh` inconsistent with the relative imports used everywhere
  else), 2 × `E741` ambiguous `l` in `linops.py:414,434`, and one real annotation bug:
  `triangular.py:461` declares `-> Int[jax.Array, "n_faces"]` for a function that returns
  and documents a **boolean mask**. Fixing it to `Bool[jax.Array, " n_faces"]` clears the
  `F821` *and* the unused-`Bool` `F401` at once. [V]
- **`lineax` is a hard dependency but is imported and never used** (`linops.py:19`). Drop it
  from `pyproject.toml`. `igl` and `numpy` are unused in several modules but remain genuine
  deps via `mesh.py`/`triangular.py`. [V]
- **Docstrings that contradict the code**: `geometry.py:493,526` ("NaN for boundary
  halfedges" — the code sets 0); `geometry.py:64` ("in 2d this returns +/-1" — B7);
  `geometry.py:424` ("uses cross product" — it doesn't); `algorithms.py:76` (documents a
  scalar max angle, returns all three); `algorithms.py:30` (return annotation is an array,
  function returns a 3-tuple); `mesh.py:156` (references a `GeomHeMesh` class that doesn't
  exist); `simulation.py:42` ("step_fn receives consecutive pairs" — it gets one scalar);
  `simulation.py:69` (claims early stopping; `on_chunk`'s return value is discarded);
  `triangular.py:33` (documents inf-sentinel restoration the function doesn't implement).
- **`mesh.py:221/231`**: `**Static methods**` heading appears twice in the `HeMesh` docstring;
  `iterate_around_vertex`/`save` are filed under "Class methods" but are instance methods.
- **Undocumented conventions** that users will get wrong: the Laplacian's sign (stated only
  in notebook markdown), which area convention `normalize=True` uses, the SVK
  `alpha`/`beta` → Lamé conversion (it's `alpha = λ/8, beta = μ/8`), which vertex is `a` in
  `get_metric`'s edge basis, and the neo-Hookean-vs-SVK inversion-barrier tradeoff.
- **`get_helfrich_energy`'s args tuple is `(hemesh, H0, kappa_H, kappa_K)`** while the
  docstring formula reads `kappa_H/2 * (H - H0)^2` — the reading order invites swapping the
  first two. (I made exactly this mistake while verifying.) Worth a note or a reorder.
- **Naming**: `get_*` (geometry, periodic) vs `compute_*` (linops) vs bare
  (`mass_matrix_sparse`) with no stated rule; `linops` calls the periodic displacement
  callback `distance_function` while `periodic.py` calls it `displacement_fn` and notebook
  05b explicitly warns it is *not* a distance.
- **Style-guide violations**: `TriMesh.set_voronoi` mutates in place and returns `None`;
  `generate_ginibre_points`/`generate_poisson_points` use global `np.random` (irreproducible,
  not vmappable) while everything else uses `jax.random` keys; `elastic.py:311-347`'s four
  helpers are the only functions in the module with no type hints at all.
- **Packaging hygiene**: `__version__` still `0.0.2`; stale `dist/triangulax-0.0.2.*` from
  April is still present (`twine upload dist/*` would trip on it); `MANIFEST.in` references
  a non-existent `settings.ini` and `CONTRIBUTING.md`; PyPI `keywords` are still the nbdev
  boilerplate `['nbdev','jupyter','notebook','python']`; a stray
  `Screenshot 2026-02-10….png` is tracked at repo root; README and `nbs/index.ipynb` list a
  module named `elasticity` (it's `elastic`); the README quickstart reads
  `test_meshes/disk.obj`, which a pip user does not have. [V]

---

## E. Omissions & design

> **[IMPLEMENTED IN COMMITS b4a9d12 (features) AND d6a4131 (design)]** Features: E2 area/volume constraint energies, E3 normal derivative (+sparse), E4 periodic sparse Laplacian/mass matrix (+ periodic robust Voronoi areas), E5 periodic oriented triangle areas, E6 topological summary helpers (module-level functions), E8 geodesic curvature (Gauss-Bonnet verified to <1e-13). Design: `GeomMesh` no longer duplicates element counts; `read_obj` defaults to `dim=3`; `get_adjacent_vertex_indices` removed; `TriMesh` documented as a non-pytree I/O container; `MeshReindexMap` and `Mesh` kept. **Not implemented by request:** E9 (padded traversal — declined in favour of gather/scatter), E10 (spatial acceleration), and a periodic FE gradient.

**Genuine gaps for the stated scope** (ranked by value):

1. **Edge split** — you have flip, collapse, and smoothing; split is the one primitive
   blocking a standard Botsch–Kobbelt `isotropic_remesh` loop, which would then be a thin
   wrapper over what already exists. Already acknowledged in notebook 03 cell 50.
2. **Area / volume constraint energies.** `get_helfrich_energy` is essentially unusable for
   its canonical application (vesicles, RBC shapes) without them — the Helfrich functional
   is scale-invariant at `H0 = 0`, which your own notebook asserts, so it has no length
   scale under minimization. `geom.get_volume` and `geom.get_area` already exist; this is
   two ~5-line functions.
3. **Normal derivative / flux across edge** — item 3 in your own
   `_features_to_be_implemented.txt`, and the natural companion to the existing divergence.
4. **Periodic counterparts for the sparse operators.** `compute_periodic_cotan_laplace`
   exists in isolation: no periodic mass matrix, no periodic sparse Laplacian, no periodic
   FE gradient. A user cannot assemble the notebook's own implicit-diffusion example on a
   periodic mesh — and periodic 2D tissue is the primary use case.
5. **`get_periodic_oriented_triangle_areas`.** All periodic areas go through Heron on
   min-image lengths, so they are unconditionally ≥ 0 and **orientation-blind**: inverting a
   triangle leaves the area unchanged. In a vertex model an inverted (T1-needing) triangle
   is therefore undetectable — exactly the event these simulations must catch.
6. **Topological summary helpers** — `euler_characteristic`, `genus`, `n_bdry_loops`,
   `is_closed`, `connected_components`. All one-liners over data `HeMesh` already holds,
   and they'd make `get_volume`'s precondition (B25) and a Gauss–Bonnet test checkable.
7. **`halfedgeVectorsInFace/InVertex`** — computed inline twice inside the transport
   functions and never exposed; they are the natural primitive for any tangent-vector-field
   work and would have made B6 a one-liner.
8. **Geodesic curvature at boundary vertices** — without it neither Gauss–Bonnet nor a
   sensible boundary Gaussian curvature is available on open meshes.
9. **A JAX-native one-ring traversal.** `iterate_around_vertex` is a Python `while` loop, so
   it can't be jitted or vmapped — the one place the docstring itself flags as a limitation,
   in a library whose whole premise is JAX-native. A padded
   `one_ring(max_valence) -> (n_vertices, max_valence)` with `-1` fill would fix it; notebook
   cell 28 sketches this but it was never exported.
10. **Spatial acceleration for closest-point queries** — `find_closest_faces` is brute force
    and materializes ~5× `n_points·n_faces·8` bytes (230 MB for 5 000 × 1 152). A BVH is a
    real project and is fine to defer, but the memory figure should be documented.

**Conflicts with the "reusable pure functions, avoid custom types" goal:**

- **`Mesh` (`mesh.py:581`) is vestigial** — constructed once in a demo cell, referenced
  nowhere. Dropping it now is free; after publication it is a breaking change.
- **`GeomMesh` duplicates state it cannot keep consistent.** It stores `n_vertices`/`n_hes`/
  `n_faces` as static fields that are already derivable from array shapes, then adds
  `check_compatibility` and `validate_dimensions` to detect the resulting drift — and
  `validate_dimensions` is itself wrong for batched meshes. Deriving them as properties
  would delete three fields and two methods. This is the one place a larger simplification
  is worth considering pre-1.0, precisely because it's cheap now.
- **`check_boundary` on `flip_edge`/`collapse_edge`** is a flag whose only implementation is
  a Python `assert` that cannot execute under jit (B5). Delete it; the `can_*_edge`
  predicates are the correct pattern.
- **`MeshReindexMap` is produced but never consumed** — nothing uses it to carry `GeomMesh`
  attributes through a topology change; the notebook does it by hand.
- **`TriMesh` is not a JAX pytree** (unlike `HeMesh`/`GeomMesh`) and is mutable. Fine if it's
  an I/O-only container, but the class docstring should say so.
- **`get_adjacent_vertex_indices` returns a ragged `list[jax.Array]`** — not jittable,
  vmappable, or differentiable, and requires igl at call time.
- **`read_obj`'s `dim=2` default silently collapses 3D meshes**: `read_obj("sphere.obj")`
  maps 42 vertices onto 25 distinct 2D positions with no warning. Consider defaulting to 3.

---

## F. Suggested triage order

**Must fix before publishing** (all small, all verified):
1. A1 — three `np.asarray(..., dtype=np.int64)` conversions at the igl boundaries
2. A2 — `requires-python = ">=3.10"`
3. A3 — one underscore in `linops.py:173`
4. B1/B2 — one `jnp.where` in `get_dihedral_angles` (fixes the Helfrich energy too)
5. B3/B4/B5 — one `can_flip_edge(hh, e)` in the flip scans; delete `check_boundary`
6. B6 — `get_signed_angle_between_vectors` in both transport functions
7. B8 — `jnp.allclose` → `jnp.array_equal` in `test_mesh_validity` and `HeMesh.__eq__`
8. A4 — fix the 4 failing notebooks; drop `lineax`; bump `__version__`; clear `dist/`

**Strongly recommended in the same pass** (each is a one- or two-line guard):
B7, B10, B13, B14+B15 (together), B17, B18, B19, B20, B21, plus the `ruff --fix` sweep and
the `triangular.py:461` annotation.

**The durable fix:** convert the bare `print`/expression cells in notebooks 02, 04 and 06
to `assert`s, and add one curved-open-mesh fixture plus one x64-off smoke test. That
combination would have caught six of the eight blockers above on its own.
