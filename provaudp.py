import socket
import select
import time

HOST = "192.168.1.102"
PORTS = [6699, 7788]

sockets = []
for port in PORTS:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, port))
    sockets.append(s)
    print(f"Bound on {HOST}:{port}")

print("Waiting for UDP packets...")

counts = {6699: 0, 7788: 0}
t0 = time.time()

while True:
    readable, _, _ = select.select(sockets, [], [], 2.0)
    if not readable:
        print("timeout...")
        continue

    for s in readable:
        data, addr = s.recvfrom(4096)
        port = s.getsockname()[1]
        counts[port] += 1
        print(
            f"port={port} count={counts[port]} "
            f"from={addr[0]}:{addr[1]} len={len(data)} "
            f"elapsed={time.time() - t0:.1f}s"
        )