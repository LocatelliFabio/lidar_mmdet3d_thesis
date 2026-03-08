#include <pcap.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#pragma pack(push, 1)
struct EthHdr {
    uint8_t dst[6];
    uint8_t src[6];
    uint16_t ethertype;
};

struct Ipv4Hdr {
    uint8_t ver_ihl;
    uint8_t tos;
    uint16_t total_len;
    uint16_t id;
    uint16_t frag_off;
    uint8_t ttl;
    uint8_t proto;
    uint16_t checksum;
    uint32_t saddr;
    uint32_t daddr;
};

struct UdpHdr {
    uint16_t sport;
    uint16_t dport;
    uint16_t len;
    uint16_t checksum;
};
#pragma pack(pop)

struct PacketInfo {
    uint64_t ts_ns;              // timestamp assoluto dal pcap in nanosecondi
    uint16_t dport;              // porta di destinazione UDP
    std::vector<uint8_t> payload;
};

static bool parse_udp_packet(const u_char* packet,
                             uint32_t caplen,
                             uint16_t& out_dport,
                             const uint8_t*& out_payload,
                             size_t& out_payload_len) {
    if (caplen < sizeof(EthHdr) + sizeof(Ipv4Hdr) + sizeof(UdpHdr)) {
        return false;
    }

    const EthHdr* eth = reinterpret_cast<const EthHdr*>(packet);
    if (ntohs(eth->ethertype) != 0x0800) {
        return false; // solo IPv4
    }

    const uint8_t* ip_ptr = packet + sizeof(EthHdr);
    const Ipv4Hdr* ip = reinterpret_cast<const Ipv4Hdr*>(ip_ptr);

    const uint8_t version = (ip->ver_ihl >> 4) & 0x0F;
    const uint8_t ihl = (ip->ver_ihl & 0x0F) * 4;

    if (version != 4 || ihl < 20) {
        return false;
    }

    if (ip->proto != 17) {
        return false; // solo UDP
    }

    if (caplen < sizeof(EthHdr) + ihl + sizeof(UdpHdr)) {
        return false;
    }

    const uint8_t* udp_ptr = ip_ptr + ihl;
    const UdpHdr* udp = reinterpret_cast<const UdpHdr*>(udp_ptr);

    const uint16_t udp_len = ntohs(udp->len);
    if (udp_len < sizeof(UdpHdr)) {
        return false;
    }

    const size_t payload_len = udp_len - sizeof(UdpHdr);
    const uint8_t* payload = udp_ptr + sizeof(UdpHdr);

    if (payload + payload_len > packet + caplen) {
        return false;
    }

    out_dport = ntohs(udp->dport);
    out_payload = payload;
    out_payload_len = payload_len;
    return true;
}

static bool load_pcap_into_memory(const std::string& pcap_file,
                                  std::vector<PacketInfo>& packets) {
    char errbuf[PCAP_ERRBUF_SIZE];
    pcap_t* pcap = pcap_open_offline(pcap_file.c_str(), errbuf);
    if (!pcap) {
        std::cerr << "Errore apertura pcap: " << errbuf << "\n";
        return false;
    }

    const int linktype = pcap_datalink(pcap);
    if (linktype != DLT_EN10MB) {
        std::cerr << "Linktype non supportato: " << linktype << " (serve Ethernet)\n";
        pcap_close(pcap);
        return false;
    }

    packets.clear();
    packets.reserve(100000);

    struct pcap_pkthdr* header = nullptr;
    const u_char* packet = nullptr;

    while (true) {
        const int rc = pcap_next_ex(pcap, &header, &packet);
        if (rc == 1) {
            uint16_t dport = 0;
            const uint8_t* payload = nullptr;
            size_t payload_len = 0;

            if (!parse_udp_packet(packet, header->caplen, dport, payload, payload_len)) {
                continue;
            }

            uint64_t ts_ns =
                static_cast<uint64_t>(header->ts.tv_sec) * 1000000000ULL +
                static_cast<uint64_t>(header->ts.tv_usec) * 1000ULL;

            PacketInfo info;
            info.ts_ns = ts_ns;
            info.dport = dport;
            info.payload.assign(payload, payload + payload_len);

            packets.push_back(std::move(info));
        } else if (rc == -2) {
            break; // EOF
        } else if (rc == -1) {
            std::cerr << "Errore lettura pcap: " << pcap_geterr(pcap) << "\n";
            pcap_close(pcap);
            return false;
        }
    }

    pcap_close(pcap);
    return true;
}

static void replay_fast(int sock, const std::vector<PacketInfo>& packets) {
    sockaddr_in dst{};
    dst.sin_family = AF_INET;
    inet_pton(AF_INET, "127.0.0.1", &dst.sin_addr);

    uint64_t sent = 0;

    for (const auto& pkt : packets) {
        dst.sin_port = htons(pkt.dport);

        ssize_t rc = sendto(sock,
                            pkt.payload.data(),
                            pkt.payload.size(),
                            0,
                            reinterpret_cast<sockaddr*>(&dst),
                            sizeof(dst));
        if (rc >= 0) {
            sent++;
        }
    }

    std::cout << "Pacchetti UDP inviati: " << sent << "\n";
}

static void replay_timed(int sock, const std::vector<PacketInfo>& packets) {
    if (packets.empty()) {
        std::cout << "Nessun pacchetto da inviare\n";
        return;
    }

    sockaddr_in dst{};
    dst.sin_family = AF_INET;
    inet_pton(AF_INET, "127.0.0.1", &dst.sin_addr);

    const uint64_t first_ts = packets.front().ts_ns;
    const auto start = std::chrono::steady_clock::now();

    uint64_t sent = 0;

    for (const auto& pkt : packets) {
        uint64_t rel_ns = pkt.ts_ns - first_ts;
        auto target = start + std::chrono::nanoseconds(rel_ns);

        std::this_thread::sleep_until(target);

        dst.sin_port = htons(pkt.dport);

        ssize_t rc = sendto(sock,
                            pkt.payload.data(),
                            pkt.payload.size(),
                            0,
                            reinterpret_cast<sockaddr*>(&dst),
                            sizeof(dst));
        if (rc >= 0) {
            sent++;
        }
    }

    std::cout << "Pacchetti UDP inviati: " << sent << "\n";
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Uso: " << argv[0] << " file.pcap --fast|--timed\n";
        return 1;
    }

    const std::string pcap_file = argv[1];
    const std::string mode = argv[2];

    std::vector<PacketInfo> packets;
    std::cout << "Caricamento pcap in memoria...\n";

    if (!load_pcap_into_memory(pcap_file, packets)) {
        return 1;
    }

    std::cout << "Pacchetti UDP caricati: " << packets.size() << "\n";
    if (!packets.empty()) {
        std::cout << "Primo timestamp ns: " << packets.front().ts_ns << "\n";
        std::cout << "Ultimo timestamp ns: " << packets.back().ts_ns << "\n";
    }

    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) {
        perror("socket");
        return 1;
    }

    int sndbuf = 4 * 1024 * 1024;
    if (setsockopt(sock, SOL_SOCKET, SO_SNDBUF, &sndbuf, sizeof(sndbuf)) != 0) {
        perror("setsockopt(SO_SNDBUF)");
    }

    if (mode == "--fast") {
        replay_fast(sock, packets);
    } else if (mode == "--timed") {
        replay_timed(sock, packets);
    } else {
        std::cerr << "Modalità non valida: " << mode << "\n";
        std::cerr << "Usa --fast oppure --timed\n";
        close(sock);
        return 1;
    }

    close(sock);
    return 0;
}