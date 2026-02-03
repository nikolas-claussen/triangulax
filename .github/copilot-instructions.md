#  Copilot Instructions (Repository-wide)

Generally, be as concise as reasonably possible. Avoid humor or emojis.

## Project context

The goal of this project is to provide data-structures for triangular meshes and functions for geometry processing based on JAX and fully compatible with JAX's just-in-time compilation and automatic differentiation. 

The current use case is simulations for the mechanics of 2D tissues ([Active tension networks](https://www.pnas.org/doi/10.1073/pnas.2321928121), [area-perimeter vertex model](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.6.021011), non-confluent tissues like in [Kim, 2021](https://www.nature.com/articles/s41567-021-01215-1), etc) compatible with automatic differentiation/JAX. The ultimate goal is to use gradient-based optimization to identify models for tissue dynamics that produce certain behaviors of interest.

However, the library is intended to be (eventually) be more general-purpose and applicable to other geometry processing tasks on triangular meshes. 

## Coding standards (applies to all Python)

- Use **Python 3.10+**.
- Always include **type hints** on function signatures and variables where practical. Use `jaxtyping` for array type hints (include informative names for array dimensions).
- Use **docstrings** (NumPy style). Include units and parameter domains when relevant.
- Follow standard **PEP8** style guidelines. Lint with `ruff`.
- Where possible, use a **functional programming style**: avoid mutable state, side effects, and in-place modifications.
- Develoment is done in **Jupyter Notebooks** (see below). Separate cells for defining functions/classes and for running code. Below a cell that defines a function/class, include a test cell that runs basic tests or examples of usage.

## External libraries

- Avoid using 3rd party libraries unless necessary beyond the ones already in the project (see `environment.yml`).
- Use **JAX** for numerical computations. Use `jax.numpy` instead of standard `numpy`.
- Use **igl** for all geometrry processing not done in JAX (loading and saving meshes etc), and to test that JAX computations are correct (compare to igl results).
- Use `equinox` for neural network functionality if needed. `diffrax`, `lineax` and `optimistix` for numerics

## Packaging & docs (nbdev)

- All code is developed in **Jupyter Notebooks** in the `nbdocs` folder.
- Use **nbdev** to export code via the `ndbdev_export` command. Do not edit the code files in `triangulax/` directly. Cells to be exported should be marked with `#| export` at the top. Cells with time-consuming computations should be marked with `#| notest`.
- Documentation webpage is in the `docs` folder. To generate documentation, use `nbdev_docs` and `nbdev_readme`. Nbdev places docs in the `_docs` folder. To update the documentation webpage, delete the old `docs` folder, run the nbdev commands, then move `_docs` to `docs`.
- Use the `triangulax` conda environment for development.
- Name notebooks with a `00_`, `01_`, etc prefix to indicate order. If a notebook exports code, a cell at the start with `#| default_exp MODULE_NAME` should be included.

## Performance & numerics

- Prefer vectorized JAX operations. Use `jax.numpy` instead of standard `numpy` where possible.
- Use `jax.vmap` for batching operations over data points or other dimensions.
- Instead of in-place array modifications, use JAX's `x = x.at[idx].set(y)` syntax.
- If you need example or test data, use the `nbs/test_meshes/` directory. Request me if you need extra test data.
