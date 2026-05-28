"""Temporary — proves CI fails when tests fail. Removed in the next commit."""


def test_ci_must_go_red_then_delete_this_file() -> None:
    assert False, "intentional CI sanity-check failure; next commit reverts"
