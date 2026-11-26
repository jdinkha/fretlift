import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from fretlift.view import Handler


def test_viewer_lists_and_serves_only_tab_files(tmp_path):
    (tmp_path / "song.musicxml").write_text("<score/>")
    (tmp_path / "song.wav").write_bytes(b"RIFF")
    (tmp_path.parent / "secret.musicxml").write_text("no")
    Handler.default_dir = tmp_path
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"

    def get(path, host=None):
        req = urllib.request.Request(base + path, headers={"Host": host} if host else {})
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    try:
        code, body = get("/api/list")
        assert code == 200 and json.loads(body)["files"] == ["song.musicxml"]
        assert get(f"/api/file?dir={tmp_path}&name=song.musicxml") == (200, b"<score/>")
        assert get(f"/api/file?dir={tmp_path}&name=song.wav")[0] == 404
        assert get(f"/api/file?dir={tmp_path}&name=../secret.musicxml")[0] == 404
        assert get("/api/list?dir=/no/such/folder")[0] == 404
        assert get("/api/list", host="evil.example:80")[0] == 403
        assert get("/")[0] == 200
    finally:
        server.shutdown()
