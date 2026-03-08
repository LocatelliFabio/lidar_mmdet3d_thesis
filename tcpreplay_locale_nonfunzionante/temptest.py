import socket
import select

ports = [6699, 7788]
socks = []

for port in ports:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("0.0.0.0", port))
    socks.append(s)
    print(f"bound su UDP {port}")

while True:
    r, _, _ = select.select(socks, [], [])
    for s in r:
        data, addr = s.recvfrom(4096)
        print(f"porta {s.getsockname()[1]}: {len(data)} byte da {addr}")