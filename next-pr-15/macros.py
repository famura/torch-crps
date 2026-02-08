import logging
from contextlib import redirect_stdout
from os import devnull
from pathlib import Path

from genbadge.utils_coverage import get_coverage_badge, get_coverage_stats
from genbadge.utils_junit import get_test_stats, get_tests_badge
from git_changelog import build_and_render
from mkdocs_macros.plugin import MacrosPlugin
from piplicenses import create_output_string, create_parser

PIP_LICENSES_CLI_ARGS = (
    "--format",
    "markdown",
    "--ignore-packages",
    "torch-crps",
    "--with-authors",
    "--with-description",
    "--with-urls",
)


def define_env(env: MacrosPlugin) -> None:
    """Define macros for the mkdocs environment.

    They can be used in the markdown files like this `{{ my_macro() }}`.

    Args:
        env: The mkdocs macros environment.
    """

    @env.macro
    def make_changelog() -> str:
        with redirect_stdout(open(devnull, mode="w", encoding="utf-8")):
            _, rendered = build_and_render(repository=".", template="keepachangelog", convention="conventional")
        return rendered

    @env.macro
    def make_third_party_license_summary() -> str:
        return create_output_string(args=create_parser().parse_args(args=(*PIP_LICENSES_CLI_ARGS, "--summary")))

    @env.macro
    def make_third_party_license_table() -> str:
        return create_output_string(args=create_parser().parse_args(args=PIP_LICENSES_CLI_ARGS))

    @env.macro
    def make_readme() -> str:
        """Return a version of the original project readme.md with updated paths."""
        with open("readme.md", encoding="utf-8") as file:
            lines = file.readlines()

        def _replace_url(line: str, pattern: str, replace_pattern: str) -> str:
            if pattern in line:
                line = line.replace(pattern, replace_pattern)
                if not Path(line).suffix:
                    line = line[:-1] + ".md\n"
            return line

        lines = [_replace_url(line, pattern="docs/", replace_pattern="./") for line in lines]
        lines = [
            _replace_url(
                line,
                pattern="https://famura.github.io/torch-crps/latest/",
                replace_pattern="./",
            )
            for line in lines
        ]
        # Keep the absolute URLs for the badge images that are generated during build.
        lines = [
            line.replace(
                "./exported/coverage/badge.svg",
                "https://famura.github.io/torch-crps/latest/exported/coverage/badge.svg",
            )
            if "./exported/coverage/badge.svg" in line
            else line
            for line in lines
        ]
        lines = [
            line.replace(
                "./exported/tests/badge.svg",
                "https://famura.github.io/torch-crps/latest/exported/tests/badge.svg",
            )
            if "./exported/tests/badge.svg" in line
            else line
            for line in lines
        ]
        return "".join(lines)

    @env.macro
    def make_api_reference() -> str:
        """Return the API reference."""
        module_path = Path("torch_crps")
        rendered_lines = (
            f"::: {'.'.join(path.with_suffix('').parts)}"
            for path in sorted(module_path.rglob("*.py"))
            if "__init__.py" not in str(path)
        )
        return "\n".join(rendered_lines)


def on_post_build(env: MacrosPlugin) -> None:
    """Create additional files."""
    site_dir = Path(env.conf["site_dir"])

    # Coverage badge
    fn = Path("coverage.xml")
    if fn.exists():
        badge = get_coverage_badge(cov_stats=get_coverage_stats(coverage_xml_file=str(fn)), left_txt="Coverage")
        badge.write_to(site_dir / "exported" / "coverage" / "badge.svg")
    else:
        logging.warning(f"File {fn} not found. Make sure to run the tests (with coverage reporting) first.")

    # Tests badge
    fn = Path("pytest.xml")
    if fn.exists():
        badge = get_tests_badge(test_stats=get_test_stats(junit_xml_file=str(fn)))
        badge.write_to(site_dir / "exported" / "tests" / "badge.svg")
    else:
        logging.warning(f"File {fn} not found. Make sure to run the tests first.")
