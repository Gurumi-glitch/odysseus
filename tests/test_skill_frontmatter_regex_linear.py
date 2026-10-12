"""Frontmatter key/list lines with long whitespace runs still parse the same."""
from services.memory.skill_format import parse_frontmatter


def test_padded_scalar_and_list_values_parse_stripped():
    pad = " " * 5000
    text = f"---\nname:{pad}demo{pad}\ntags:{pad}\n  -{pad}a{pad}\n  - b\n---\nbody"
    fm, body = parse_frontmatter(text)
    assert fm["name"] == "demo"
    assert fm["tags"] == ["a", "b"]
    assert body == "body"
