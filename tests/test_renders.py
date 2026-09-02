import io
import tarfile
from pathlib import Path

import pytest

from magiccut.data.renders import parse_render_name, safe_extract_tar


def test_parse_render_name() -> None:
    parsed = parse_render_name("abc123_7_7_front_medium.png")
    assert parsed.uid == "abc123"
    assert parsed.part_id == 7
    assert parsed.representative_id == 7
    assert parsed.view == "front"
    assert parsed.size == "medium"


def test_safe_extract_rejects_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        member = tarfile.TarInfo("../escape.png")
        payload = b"not-an-image"
        member.size = len(payload)
        handle.addfile(member, io.BytesIO(payload))
    with pytest.raises(ValueError, match="Unsafe archive path"):
        safe_extract_tar(archive, tmp_path / "out")
