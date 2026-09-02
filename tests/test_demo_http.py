from fastapi.testclient import TestClient

import scripts.serve_demo as server


class FakeEngine:
    def meshes(self):
        return [{"uid": "mesh", "part_count": 2, "part_ids": [1, 2]}]

    def start(self, uid, query_part):
        return {"uid": uid, "query_part": query_part, "session_id": "session"}

    def feedback(self, session_id, part_id, positive):
        return {"session_id": session_id, "part_id": part_id, "positive": positive}

    def reset(self, session_id):
        return {"session_id": session_id, "click_count": 0}


def test_demo_api_contract(monkeypatch):
    monkeypatch.setattr(server, "engine", FakeEngine())
    client = TestClient(server.app)
    assert client.get("/").status_code == 200
    assert client.get("/api/meshes").json()[0]["part_ids"] == [1, 2]
    assert client.post("/api/start", json={"uid": "mesh", "query_part": 2}).json()["query_part"] == 2
    assert client.post(
        "/api/feedback", json={"session_id": "session", "part_id": 1, "positive": False}
    ).json()["positive"] is False
    assert client.post("/api/reset", json={"session_id": "session"}).json()["click_count"] == 0
