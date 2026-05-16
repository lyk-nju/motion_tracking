"""Real runtime consumer smoke: SocketFloodNetSource instantiation + post_step()."""

import socket, struct, json, threading, time, numpy as np
from common.utils import DictToClass

OBS_JOINT_NAMES = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]
DATASET_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]


class _MockPolicy:
    def __init__(self):
        self.obs_joint_names = OBS_JOINT_NAMES
        self.dataset_joint_names = DATASET_JOINT_NAMES
        self.ref_joint_pos = None; self.ref_root_quat = None; self.ref_root_pos = None
        self.ref_len = 0; self.ref_idx = 0
        self.current_name = "default"; self.current_done = True
        self.appended = []

    def append_ref_frames(self, frames):
        self.appended.append(frames)
        if self.ref_joint_pos is None:
            self.ref_joint_pos = frames["joint_pos"].copy()
            self.ref_root_quat = frames["root_quat"].copy()
            self.ref_root_pos = frames["root_pos"].copy()
        else:
            self.ref_joint_pos = np.concatenate([self.ref_joint_pos, frames["joint_pos"]], 0)
            self.ref_root_quat = np.concatenate([self.ref_root_quat, frames["root_quat"]], 0)
            self.ref_root_pos = np.concatenate([self.ref_root_pos, frames["root_pos"]], 0)
        self.ref_len = self.ref_joint_pos.shape[0]
        self.current_done = self.ref_idx >= self.ref_len - 1

    def read_ref_tail_state(self):
        if self.ref_len > 0:
            return {"joint_pos": self.ref_joint_pos[-1], "root_quat": self.ref_root_quat[-1],
                    "root_pos": self.ref_root_pos[-1]}
        return {"joint_pos": np.zeros(29), "root_quat": np.array([1,0,0,0]),
                "root_pos": np.array([0,0,0.78])}


def _make_chunk_msg(sid="s1", cid="c1", n=6):
    dof = np.zeros((n, 29), dtype=np.float32); dof[:,0] = 0.1
    root_pos = np.zeros((n, 3), dtype=np.float32)
    root_rot = np.tile(np.array([[0,0,0,1]], dtype=np.float32), (n, 1))
    payload = {"chunk_id": cid, "start_time": 0.0, "fps": 30,
               "root_pos": root_pos.tolist(), "root_rot": root_rot.tolist(),
               "dof_pos": dof.tolist(),
               "local_body_pos": np.zeros((n, 30, 3)).tolist(),
               "local_body_rot": np.tile([[[0,0,0,1]]], (n, 30, 1)).tolist(),
               "body_names": [f"b{i}" for i in range(30)],
               "joint_names": DATASET_JOINT_NAMES}
    msg = {"type": "chunk", "session_id": sid, "chunk_id": cid,
           "overlap_frames": 4, "payload": payload}
    data = json.dumps(msg).encode("utf-8")
    return struct.pack(">I", len(data)) + data


def _default_clip():
    return {"name": "default",
            "joint_pos": np.zeros((1, 29), dtype=np.float32).tolist(),
            "root_pos": np.zeros((1, 3), dtype=np.float32).tolist(),
            "root_quat": np.array([[1, 0, 0, 0]], dtype=np.float32).tolist()}


def test_socket_source_instantiation():
    """SocketFloodNetSource can be instantiated."""
    from motion_sources import SocketFloodNetSource
    cfg = DictToClass({"socket_host": "127.0.0.1", "socket_port": 15600,
                       "motions": [], "motion_clips": [_default_clip()],
                       "motion_source": "socket_floodnet"})
    source = SocketFloodNetSource(_MockPolicy(), cfg)
    assert source.socket_host == "127.0.0.1" and source.socket_port == 15600


def test_socket_source_drain_messages():
    """post_step() drains chunks from TCP server into policy ref buffer."""
    from motion_sources import SocketFloodNetSource

    port = 15601
    ready = threading.Event()
    barrier = threading.Event()

    def _server():
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", port)); srv.listen(1); ready.set(); srv.settimeout(3.0)
        try:
            conn, _ = srv.accept()
            conn.sendall(_make_chunk_msg("s1", "c1", 6))
            conn.sendall(_make_chunk_msg("s1", "c2", 8))
            barrier.wait(timeout=2.0)  # keep conn open until drain done
            conn.close()
        except socket.timeout: pass
        srv.close()

    threading.Thread(target=_server, daemon=True).start()
    ready.wait(timeout=2.0)
    time.sleep(0.1)

    cfg = DictToClass({"socket_host": "127.0.0.1", "socket_port": port,
                       "motions": [], "motion_clips": [_default_clip()],
                       "motion_source": "socket_floodnet"})
    policy = _MockPolicy()
    source = SocketFloodNetSource(policy, cfg)

    source.post_step()  # drains messages via socket
    barrier.set()  # signal server to close
    assert len(policy.appended) == 2, f"expected 2 chunks, got {len(policy.appended)}"
    assert policy.ref_len == 14  # 6 + 8 frames
    source.deactivate()


def test_motion_source_config_accepts_socket_floodnet():
    """motion_source: socket_floodnet is accepted as valid config value."""
    cfg = DictToClass({"motion_source": "socket_floodnet", "future_steps": [0,1,2,-1],
                       "dataset_joint_names": DATASET_JOINT_NAMES,
                       "motions": [], "motion_clips": []})
    ms = str(cfg.motion_source).strip().lower()
    assert ms in ("udp", "vr", "floodnet", "socket_floodnet")


# ---- Consumer-side timeout / disconnect smoke (Task 016) ----


def _drain_until(source, expected: int, max_attempts: int = 20) -> int:
    """Retry _drain_messages until expected chunks received or attempts exhausted."""
    import time as time_mod
    total = 0
    for _ in range(max_attempts):
        total += source._drain_messages()
        if total >= expected:
            break
        time_mod.sleep(0.05)
    return total


def test_socket_source_disconnect_reason_on_server_close():
    """Consumer disconnect_reason is set when server closes connection mid-session."""
    import time as time_mod
    from motion_sources import SocketFloodNetSource

    port = 15610
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(1)
    srv.settimeout(2.0)

    server_ready = threading.Event()

    def _server():
        server_ready.set()
        try:
            conn, _ = srv.accept()
            conn.settimeout(2.0)
            conn.sendall(_make_chunk_msg("s1", "c1", 4))
            time_mod.sleep(0.3)
            conn.close()
        except socket.timeout:
            pass

    threading.Thread(target=_server, daemon=True).start()

    cfg = DictToClass({"socket_host": "127.0.0.1", "socket_port": port,
                       "motions": [], "motion_clips": [_default_clip()],
                       "motion_source": "socket_floodnet"})
    policy = _MockPolicy()
    source = SocketFloodNetSource(policy, cfg)

    server_ready.wait(timeout=2.0)
    # Connect + receive with retries (timing-tolerant)
    total = _drain_until(source, 1, max_attempts=10)
    assert total == 1, f"expected 1 chunk, got {total}"
    assert source.connected

    time_mod.sleep(0.4)
    n = source._drain_messages()
    assert n == 0
    assert not source.connected
    assert source.disconnect_reason != "", "disconnect_reason should be non-empty after server close"

    source.deactivate()
    srv.close()


def test_socket_source_no_reconnect_when_server_gone():
    """After disconnect, source stays disconnected and does not hang."""
    import time as time_mod
    from motion_sources import SocketFloodNetSource

    port = 15611
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(1)
    srv.settimeout(2.0)

    server_ready = threading.Event()

    def _server():
        server_ready.set()
        try:
            conn, _ = srv.accept()
            conn.settimeout(2.0)
            conn.sendall(_make_chunk_msg("s1", "c1", 4))
            time_mod.sleep(0.2)
            conn.close()
        except socket.timeout:
            pass
        srv.close()

    threading.Thread(target=_server, daemon=True).start()

    cfg = DictToClass({"socket_host": "127.0.0.1", "socket_port": port,
                       "motions": [], "motion_clips": [_default_clip()],
                       "motion_source": "socket_floodnet"})
    policy = _MockPolicy()
    source = SocketFloodNetSource(policy, cfg)

    server_ready.wait(timeout=2.0)
    total = _drain_until(source, 1, max_attempts=10)
    assert total == 1

    # Server is gone — multiple drain calls should return 0 without hanging
    time_mod.sleep(0.3)
    for _ in range(5):
        n = source._drain_messages()
        assert n == 0, "should get 0 chunks after server gone"
    assert not source.connected
    source.deactivate()


def test_socket_source_drain_when_no_server():
    """When no server is listening, drain returns 0 and disconnect_reason is set."""
    from motion_sources import SocketFloodNetSource

    port = 15612
    cfg = DictToClass({"socket_host": "127.0.0.1", "socket_port": port,
                       "motions": [], "motion_clips": [_default_clip()],
                       "motion_source": "socket_floodnet"})
    policy = _MockPolicy()
    source = SocketFloodNetSource(policy, cfg)

    assert not source.connected
    assert source.disconnect_reason == ""

    n = source._drain_messages()
    assert n == 0
    assert not source.connected
    assert source.disconnect_reason != "", \
        f"disconnect_reason should be set on connection refused, got: {source.disconnect_reason!r}"
    source.deactivate()


def test_socket_source_reconnect_clears_disconnect_reason():
    """After deactivate + fresh connect, disconnect_reason is cleared."""
    import time as time_mod
    from motion_sources import SocketFloodNetSource

    port = 15613

    server_ready1 = threading.Event()

    def _server1():
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", port))
        srv.listen(1)
        srv.settimeout(2.0)
        server_ready1.set()
        try:
            conn, _ = srv.accept()
            conn.settimeout(2.0)
            conn.sendall(_make_chunk_msg("s1", "c1", 4))
            time_mod.sleep(0.3)
            conn.close()
        except socket.timeout:
            pass
        srv.close()

    threading.Thread(target=_server1, daemon=True).start()

    cfg = DictToClass({"socket_host": "127.0.0.1", "socket_port": port,
                       "motions": [], "motion_clips": [_default_clip()],
                       "motion_source": "socket_floodnet"})
    policy = _MockPolicy()
    source = SocketFloodNetSource(policy, cfg)

    server_ready1.wait(timeout=2.0)
    total1 = _drain_until(source, 1, max_attempts=10)
    assert total1 == 1
    assert source.connected

    # Wait for server to close → detect disconnect
    time_mod.sleep(0.5)
    source._drain_messages()
    assert not source.connected
    reason_before = source.disconnect_reason
    assert reason_before != ""

    # Round 2: new server, deactivate + reconnect
    source.deactivate()
    assert not source.connected

    server_ready2 = threading.Event()

    def _server2():
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", port))
        srv.listen(1)
        srv.settimeout(2.0)
        server_ready2.set()
        try:
            conn, _ = srv.accept()
            conn.settimeout(2.0)
            conn.sendall(_make_chunk_msg("s1", "c2", 6))
            time_mod.sleep(0.3)
            conn.close()
        except socket.timeout:
            pass
        srv.close()

    threading.Thread(target=_server2, daemon=True).start()

    server_ready2.wait(timeout=2.0)
    total2 = _drain_until(source, 1, max_attempts=10)
    assert total2 == 1, f"reconnect should get 1 chunk, got {total2}"
    assert source.connected
    assert source.disconnect_reason == "", \
        f"disconnect_reason should be cleared on reconnect, got: {source.disconnect_reason!r}"
    source.deactivate()


if __name__ == "__main__":
    import sys; sys.path.insert(0, "sim2real/src")
    test_socket_source_instantiation(); print("OK: instantiation")
    test_socket_source_drain_messages(); print("OK: drain messages")
    test_motion_source_config_accepts_socket_floodnet(); print("OK: config")
    print("\nAll runtime consumer smoke passed!")
