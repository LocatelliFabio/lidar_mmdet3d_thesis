#include <pcap.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <sched.h>
#include <sys/time.h>
#include <cerrno>
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
    uint64_t ts_ns;
    uint16_t dport;
    std::vector<uint8_t> payload;
};

static inline uint64_t now_ns() {
    using namespace std::chrono;
    return duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();
}

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
        return false;
    }

    const uint8_t* ip_ptr = packet + sizeof(EthHdr);
    const Ipv4Hdr* ip = reinterpret_cast<const Ipv4Hdr*>(ip_ptr);

    const uint8_t version = (ip->ver_ihl >> 4) & 0x0F;
    const uint8_t ihl = (ip->ver_ihl & 0x0F) * 4;

    if (version != 4 || ihl < 20) {
        return false;
    }

    if (ip->proto != 17) {
        return false;
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
            break;
        } else if (rc == -1) {
            std::cerr << "Errore lettura pcap: " << pcap_geterr(pcap) << "\n";
            pcap_close(pcap);
            return false;
        }
    }

    pcap_close(pcap);
    return true;
}

static bool set_realtime_priority(int priority = 80) {
    sched_param sp{};
    sp.sched_priority = priority;
    if (sched_setscheduler(0, SCHED_FIFO, &sp) != 0) {
        return false;
    }
    return true;
}

static bool pin_to_cpu0() {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    return sched_setaffinity(0, sizeof(cpuset), &cpuset) == 0;
}

static void replay_fast(int sock, const std::vector<PacketInfo>& packets) {
    sockaddr_in dst{};
    dst.sin_family = AF_INET;
    inet_pton(AF_INET, "127.0.0.1", &dst.sin_addr);

    uint64_t sent = 0;

    for (const auto& pkt : packets) {
        dst.sin_port = htons(pkt.dport);

        const ssize_t rc = sendto(sock,
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

static void precise_wait_until(uint64_t target_ns, uint64_t spin_threshold_ns) {
    while (true) {
        const uint64_t now = now_ns();
        if (now >= target_ns) {
            return;
        }

        const uint64_t remaining = target_ns - now;

        if (remaining > spin_threshold_ns) {
            const uint64_t sleep_ns = remaining - spin_threshold_ns;
            timespec ts{};
            ts.tv_sec = static_cast<time_t>(sleep_ns / 1000000000ULL);
            ts.tv_nsec = static_cast<long>(sleep_ns % 1000000000ULL);
            while (clock_nanosleep(CLOCK_MONOTONIC, 0, &ts, &ts) == EINTR) {
            }
        } else {
            while (now_ns() < target_ns) {
#if defined(__x86_64__) || defined(__i386__)
                __builtin_ia32_pause();
#else
                std::this_thread::yield();
#endif
            }
            return;
        }
    }
}

static void replay_timed_precise(int sock,
                                 const std::vector<PacketInfo>& packets,
                                 uint64_t spin_threshold_ns) {
    if (packets.empty()) {
        std::cout << "Nessun pacchetto da inviare\n";
        return;
    }

    sockaddr_in dst{};
    dst.sin_family = AF_INET;
    inet_pton(AF_INET, "127.0.0.1", &dst.sin_addr);

    const uint64_t first_ts = packets.front().ts_ns;
    const uint64_t start_ns = now_ns();

    uint64_t sent = 0;
    uint64_t max_late_ns = 0;

    for (const auto& pkt : packets) {
        const uint64_t rel_ns = pkt.ts_ns - first_ts;
        const uint64_t target_ns = start_ns + rel_ns;

        precise_wait_until(target_ns, spin_threshold_ns);

        const uint64_t send_begin_ns = now_ns();
        if (send_begin_ns > target_ns) {
            const uint64_t late_ns = send_begin_ns - target_ns;
            if (late_ns > max_late_ns) {
                max_late_ns = late_ns;
            }
        }

        dst.sin_port = htons(pkt.dport);

        const ssize_t rc = sendto(sock,
                                  pkt.payload.data(),
                                  pkt.payload.size(),
                                  0,
                                  reinterpret_cast<sockaddr*>(&dst),
                                  sizeof(dst));
        if (rc >= 0) {
            sent++;
        }
    }

    const uint64_t total_ns = now_ns() - start_ns;

    std::cout << "Pacchetti UDP inviati: " << sent << "\n";
    std::cout << "Durata replay reale: " << (total_ns / 1e9) << " s\n";
    std::cout << "Ritardo massimo osservato: " << (max_late_ns / 1000.0) << " us\n";
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr
            << "Uso: " << argv[0]
            << " file.pcap --fast|--timed [--spin-us N] [--rt] [--cpu0]\n";
        return 1;
    }

    const std::string pcap_file = argv[1];
    const std::string mode = argv[2];

    uint64_t spin_threshold_ns = 50000ULL; // 50 us default
    bool enable_rt = false;
    bool enable_cpu0 = false;

    for (int i = 3; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--rt") {
            enable_rt = true;
        } else if (arg == "--cpu0") {
            enable_cpu0 = true;
        } else if (arg == "--spin-us") {
            if (i + 1 >= argc) {
                std::cerr << "--spin-us richiede un valore\n";
                return 1;
            }
            spin_threshold_ns = std::stoull(argv[++i]) * 1000ULL;
        } else {
            std::cerr << "Argomento sconosciuto: " << arg << "\n";
            return 1;
        }
    }

    std::vector<PacketInfo> packets;
    std::cout << "Caricamento pcap in memoria...\n";

    if (!load_pcap_into_memory(pcap_file, packets)) {
        return 1;
    }

    std::cout << "Pacchetti UDP caricati: " << packets.size() << "\n";
    if (!packets.empty()) {
        const double duration_s =
            static_cast<double>(packets.back().ts_ns - packets.front().ts_ns) / 1e9;
        std::cout << "Durata traccia: " << duration_s << " s\n";
    }

    if (enable_cpu0) {
        if (pin_to_cpu0()) {
            std::cout << "Affinita' CPU impostata su core 0\n";
        } else {
            std::cerr << "Impossibile impostare affinita' CPU\n";
        }
    }

    if (enable_rt) {
        if (set_realtime_priority()) {
            std::cout << "Scheduler realtime SCHED_FIFO attivato\n";
        } else {
            std::cerr << "Impossibile attivare scheduler realtime (serve sudo/cap_sys_nice)\n";
        }
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
        replay_timed_precise(sock, packets, spin_threshold_ns);
    } else {
        std::cerr << "Modalita' non valida: " << mode << "\n";
        close(sock);
        return 1;
    }

    close(sock);
    return 0;
}