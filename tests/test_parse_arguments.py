import pytest
from symbiont_cli.qdrant import SymbiontCLI
import sys


def test_parse_arguments(monkeypatch):
    test_args = [
        "symbiont_cli",
        "--loader_directory",
        "test_dir",
        "--collection_name",
        "test_collection",
        "3",
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    cli = SymbiontCLI()
    args = cli.parse_arguments()

    assert args.loader_directory == "test_dir"
    assert args.collection_name == "test_collection"
    assert args.k_value == 3
