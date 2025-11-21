
<!-- markdownlint-disable MD046 -->

# Getting Started for Developers

## Project Structure

| File / Directory          | Purpose                                                                  |
| ------------------------- | ------------------------------------------------------------------------ |
| `.github`                 | CI/CD workflow definitions and a PR template                             |
| `torch_crps`              | Project import modules                                                   |
| `docs`                    | Documentation directory (better write docs there instead of `readme.md`) |
| `tests`                   | Python module unit- & integration tests                                  |
| `.pre-commit-config.yaml` | `git` hook definitions comsumed by `pre-commit`                          |
| `license.md`              | The license in its long form                                             |
| `mkdocs.yml`              | Documentation config consumed by `mkdocs`                                |
| `pyproject.toml`          | Project information, dependencies and task runner configurations         |
| `readme.md`               | General project overview, displayed when visiting GitHub repository      |
| `uv.lock`                 | Contains the locked dependencies to exactly reproduce installations.     |

## Dependency Management & Packaging

To keep the dependencies of different projects from interfering with each other, it is highly recommended to create an
isolated python environment for every project.
We use [`uv`][uv] to address this issue.
By running `uv sync` inside the project directory, a new virtual environment is created automatically into which all
your dependencies are installed (from the `uv.lock` file).
This is different from running `pip install .` in an isolated virtual environment as this might use different dependency
versions.
Afterwards you can run any command within the virtual environment by simply calling

```sh
uv run <command>
```

### Testing

Executing

```sh
uv run pytest --cov
```

will run [pytest][pytest], compute the test [coverage][coverage] and fail if below the minimum coverage defined by the
`tool.coverage.report.fail_under` threshold in the `pyproject.toml` file.

### Documentation

The code documentation is based on [`mkdocs`][mkdocs] which converts markdown files into a nicely-rendered web-page.
In particular, we use the [`mkdocs-material`][mkdocs-material] package which offers more than just theming.
To generate documentation for different versions, [`mike`][mike] is used as a plugin within [`mkdocs`][mkdocs].

To build and develop docs on a local server, run

```sh
uv run mkdocs serve
```

To deploy the docs to the `gh-pages` remote branch, call

```sh
uv run mike deploy --push --update-aliases <version> <alias>
```

where `<alias>` may be any name alias for your version such as `latest`, `stable` or `whatever`.

The final documentation is located at:

```url
https://famura.github.io/torch-crps/<alias>
```

## Git Hooks

We use [`pre-commit`][pre-commit] to run git hooks helping you to develop high-quality code.
The hooks are configured in the `.pre-commit-config.yaml` file and executed before commit.

For instance, [`ruff`][ruff] & [`ruff-format`][ruff-format] fix the code base in-place to adhere to reasonable
coding standards.
`mypy`[mypy] & [`ruff`][ruff] lint the code for correctness.
These tools are configured via `pyproject.toml` and `.pre-commit-config.yaml` files.

!!! info "Installation"

    After you cloned this project and plan to develop in it, don't forget to install these hooks via

    ```sh
    uv run pre-commit install
    ```

??? example "Available pre-commit hooks"

    ```yaml
    --8<-- ".pre-commit-config.yaml"
    ```

## GitHub Actions

There are basic CI and CD pipelines, executed as [GitHub Actions workflow] when pushing changes or opening PR's.

??? example "Available workflows"

    === ".github/workflows/ci.yaml"

        ```yaml
        --8<-- ".github/workflows/ci.yaml"
        ```

    === ".github/workflows/cd.yaml"

        ```yaml
        --8<-- ".github/workflows/cd.yaml"
        ```

<!-- URLs -->
<!-- markdownlint-disable MD034 MD053 -->
[coverage]: https://coverage.readthedocs.io/
[GitHub Actions workflow]: https://docs.github.com/en/actions/using-workflows
[mike]: https://github.com/jimporter/mike
[mkdocs-material]: https://squidfunk.github.io/mkdocs-material/
[mkdocs]: https://www.mkdocs.org/
[mypy]: http://mypy-lang.org/
[pre-commit]: https://pre-commit.com/
[pypa-user-guide]: https://packaging.python.org/
[pyproject-intro]: https://packaging.python.org/en/latest/guides/writing-pyproject-toml/
[pyproject-spec]: https://packaging.python.org/en/latest/specifications/pyproject-toml/
[pytest]: https://docs.pytest.org/en/6.2.x/
[ruff-format]: https://docs.astral.sh/ruff/formatter/
[ruff]: https://docs.astral.sh/ruff/linter/
[semantic versioning]: https://semver.org/
[settings-secrets]: https://github.com/famura/torch-crps/settings/secrets/actions
[uv]: https://docs.astral.sh/uv/
