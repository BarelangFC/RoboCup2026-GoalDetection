#include "network.hpp"
#include <cstring>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <chrono>
#include <iostream>

UdpDispatcher::UdpDispatcher() {}
UdpDispatcher::~UdpDispatcher() { close(); }

bool UdpDispatcher::setup(const std::string& target_ip, int target_port, bool broadcast) {
    m_target_ip = target_ip;
    m_target_port = target_port;

    m_sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (m_sock < 0) {
        perror("[UDP] socket");
        return false;
    }

    int opt = 1;
    if (broadcast) {
        if (setsockopt(m_sock, SOL_SOCKET, SO_BROADCAST, &opt, sizeof(opt)) < 0) {
            perror("[UDP] setsockopt broadcast");
            ::close(m_sock);
            m_sock = -1;
            return false;
        }
    }

    std::cout << "[UDP] Target: " << target_ip << ":" << target_port
              << (broadcast ? " (broadcast)" : "") << std::endl;
    return true;
}

void UdpDispatcher::send_goal_event(int seq_num, int team_id, float x, float y) {
    if (m_sock < 0) return;

    GoalPacket pkt;
    pkt.seq = htons(seq_num);
    pkt.timestamp = htonl((uint32_t)(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count()));
    pkt.team_id = (uint8_t)team_id;

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(m_target_port);
    inet_pton(AF_INET, m_target_ip.c_str(), &addr.sin_addr);

    sendto(m_sock, &pkt, sizeof(pkt), 0,
           (struct sockaddr*)&addr, sizeof(addr));
}

void UdpDispatcher::close() {
    if (m_sock >= 0) {
        ::close(m_sock);
        m_sock = -1;
    }
}
