from thetokencompany import protect


class TestProtect:
    def test_wraps_text(self) -> None:
        assert protect("system:") == "<ttc_safe>system:</ttc_safe>"

    def test_empty_string(self) -> None:
        assert protect("") == "<ttc_safe></ttc_safe>"

    def test_multiline(self) -> None:
        result = protect("line1\nline2")
        assert result == "<ttc_safe>line1\nline2</ttc_safe>"
