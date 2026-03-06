# Analyze the pcap file and print some statistics about the UDP packets

from scapy.all import rdpcap, UDP

pcap_file = "scapy/RubyPcap500pacchetti.pcap"
pkts = rdpcap(pcap_file)

ports = {}
n_udp = 0
for p in pkts:
    if UDP in p:
        n_udp += 1
        dport = int(p[UDP].dport)
        ports[dport] = ports.get(dport, 0) + 1

print("UDP packets:", n_udp)
print("Top dports:", sorted(ports.items(), key=lambda x: -x[1])[:10])