
<!-- markdownlint-disable MD046 -->

# Installation for Developers

## Cloning the Repository

For development, you need the source code. Clone the repository by executing

```sh
git clone https://github.com/famura/torch-crps.git
```

## uv

This project is managed by [`uv`][uv], an extremely fast Python package and project manager, written in Rust.
This means, however, that [`uv`][uv] needs to be installed _before_ you can run the CLI or install the
development version.
The official [`uv` installation docs][uv-install] show several ways to install []`uv`][uv] on different platforms and
under different conditions.

??? info "Installation for the lazy ones"

    This is just a quick tip on installing uv, better see the [official uv docs][uv-install]

    === "via curl on macOS & Linux (recommended)"

        Run

        ```sh
        curl -LsSf https://astral.sh/uv/install.sh | sh
        ```

    === "via conda (not recommended)"

        ```sh
        conda activate <some-environment>
        conda install pip
        pip install pipx
        pipx install uv
        conda deactivate
        ```

### Managing Python

With [`uv`][uv], you can install a suitable python version by running

```sh
uv python install
```

to install the latest stable Python version.
See the [uv docs][uv-python] for more details.


## Actual Installation

The final project installation should be easy. Run this from the projects root level:

```sh
uv sync
```

!!! tip "No project development intended?"

    If you don't need any development setup, you can add the `--no-dev` flag to skip development dependencies.

??? failure "Computer says no…"

    In some cases, this does not work right away.
    Please create and issue such that we can collect failure cases and hints to their solution here.

    | What?             | Hint             |
    | :---------------- | :--------------- |
    | placeholder issue | placeholder hint |


<!-- URLs -->
[uv-install]: https://docs.astral.sh/uv/getting-started/installation/
[uv-python]: https://docs.astral.sh/uv/guides/install-python/
[uv]: https://docs.astral.sh/uv/
