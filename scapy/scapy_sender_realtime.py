import socket
import time
from scapy.all import PcapReader, UDP

pcap_file = "scapy/RubyPcap70000pacchetti.pcap"
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

sent = 0
prev_ts = None

print("Starting replay...")

with PcapReader(pcap_file) as pkts:
    for p in pkts:
        if UDP not in p:
            continue

        ts = float(p.time)
        if prev_ts is not None:
            delay = ts - prev_ts
            if delay > 0:
                time.sleep(delay)

        dport = int(p[UDP].dport)
        payload = bytes(p[UDP].payload)
        sock.sendto(payload, ("127.0.0.1", dport))

        sent += 1
        prev_ts = ts

print("sent udp:", sent)