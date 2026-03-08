# Ambiente scapyenv
# conda create -n scapyenv python=3.8 -y
# pip install scapy
# .../miniconda3/envs/scapyenv/bin/python scapy/scapy_sender.py

import socket
from scapy.all import rdpcap, UDP

pcap_file = "scapy/RubyPcap70000pacchetti.pcap"
pkts = rdpcap(pcap_file)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

sent = 0
#ports = {}

print(f"Starting to send {len(pkts)}...")

for p in pkts:
    if UDP in p:
        dport = int(p[UDP].dport)
        payload = bytes(p[UDP].payload)
        sock.sendto(payload, ("127.0.0.1", dport))
        sent += 1
        #ports[dport] = ports.get(dport, 0) + 1

print("sent udp:", sent)
#print("sent per dport (top10):", sorted(ports.items(), key=lambda x: -x[1])[:10])