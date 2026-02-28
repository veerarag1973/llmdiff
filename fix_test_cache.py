"""Fix malformed test_put_writes_valid_json in test_cache.py."""
f = r"d:\Sriram\llmdiff\tests\test_cache.py"
text = open(f, encoding="utf-8").read()

bad = (
    '        assert any("Failed to write" in r.message for r in caplog.records)\n'
    '\n'
    '        self, tmp_cache: ResultCache, sample_response: ModelResponse\n'
    '    def test_put_writes_valid_json(\n'
    '    ) -> None:\n'
    '        key = tmp_cache.make_key("gpt-4o", "test", 0.7, 512)\n'
    '        tmp_cache.put(key, sample_response)\n'
    '        raw = tmp_cache._entry_path(key).read_text(encoding="utf-8")\n'
    '        data = json.loads(raw)\n'
    '        assert data["model"] == "gpt-4o"\n'
    '        assert data["text"] == "Hello, world!"\n'
)

good = (
    '        assert any("Failed to write" in r.message for r in caplog.records)\n'
    '\n'
    '    def test_put_writes_valid_json(\n'
    '        self, tmp_cache: ResultCache, sample_response: ModelResponse\n'
    '    ) -> None:\n'
    '        key = tmp_cache.make_key("gpt-4o", "test", 0.7, 512)\n'
    '        tmp_cache.put(key, sample_response)\n'
    '        raw = tmp_cache._entry_path(key).read_text(encoding="utf-8")\n'
    '        data = json.loads(raw)\n'
    '        assert data["model"] == "gpt-4o"\n'
    '        assert data["text"] == "Hello, world!"\n'
)

import json  # noqa: F401 (just to keep unused import from causing issues)

if bad in text:
    text = text.replace(bad, good)
    open(f, "w", encoding="utf-8").write(text)
    print("FIXED")
else:
    print("NOT FOUND")
    idx = text.find('assert any("Failed to write"')
    if idx >= 0:
        print(repr(text[idx:idx+500]))
    else:
        print("assert any line not found either")
