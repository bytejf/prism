"""CLI 入口的单元测试。"""

from importlib.metadata import version

from typer.testing import CliRunner

from prism.cli.main import app

runner = CliRunner()


def test_version_flag() -> None:
    """测试 `prism --version` 输出包版本号。"""
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == f"prism {version('prism')}"


def test_no_args_prints_help() -> None:
    """测试不带参数时打印帮助并正常退出。"""
    result = runner.invoke(app, [])
    assert result.exit_code == 0
    assert "Prism" in result.stdout
