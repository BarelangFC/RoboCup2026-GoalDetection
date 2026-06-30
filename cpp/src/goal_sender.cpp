#include "goal_sender.hpp"
#include <cstring>
#include <iostream>
#include <chrono>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <cstring>

GoalSender::GoalSender() {}
GoalSender::~GoalSender() { close(); }

bool GoalSender::setup(const std::string& host, int port) {
    m_host = host;
    m_port = port;
    std::cout << "[SEND] Target: " << host << ":" << port << " (TCP)" << std::endl;
    return true;
}

void GoalSender::send(int team_id, int seq,
                      const std::vector<std::vector<unsigned char>>& jpegs) {
    if (jpegs.empty()) return;

    // Create socket
    m_sock = socket(AF_INET, SOCK_STREAM, 0);
    if (m_sock < 0) {
        perror("[SEND] socket");
        return;
    }

    // Connect
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(m_port);
    if (inet_pton(AF_INET, m_host.c_str(), &addr.sin_addr) <= 0) {
        perror("[SEND] inet_pton");
        ::close(m_sock); m_sock = -1;
        return;
    }

    // Set connect timeout to 2 seconds
    struct timeval tv = {2, 0};
    if (setsockopt(m_sock, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv)) < 0)
        perror("[SEND] setsockopt timeout (non-fatal)");

    if (connect(m_sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        fprintf(stderr, "[SEND] connect to %s:%d failed: %s (errno=%d)\n",
                m_host.c_str(), m_port, strerror(errno), errno);
        ::close(m_sock); m_sock = -1;
        return;
    }

    // Build packet: header + frames
    uint32_t timestamp = (uint32_t)std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    // Header: team_id(1) + seq(2) + timestamp(4) + num_frames(1) = 8 bytes
    uint8_t header[8];
    header[0] = (uint8_t)team_id;
    header[1] = (uint8_t)(seq & 0xFF);
    header[2] = (uint8_t)((seq >> 8) & 0xFF);
    header[3] = (uint8_t)(timestamp & 0xFF);
    header[4] = (uint8_t)((timestamp >> 8) & 0xFF);
    header[5] = (uint8_t)((timestamp >> 16) & 0xFF);
    header[6] = (uint8_t)((timestamp >> 24) & 0xFF);
    header[7] = (uint8_t)jpegs.size();

    size_t total_sent = 0;
    ssize_t n;

    // Send header
    while (total_sent < sizeof(header)) {
        n = ::send(m_sock, header + total_sent, sizeof(header) - total_sent, 0);
        if (n <= 0) { perror("[SEND] header"); break; }
        total_sent += n;
    }

    // Send each frame
    for (size_t fi = 0; fi < jpegs.size(); fi++) {
        uint32_t frame_size = (uint32_t)jpegs[fi].size();
        uint8_t size_buf[4];
        size_buf[0] = (uint8_t)(frame_size & 0xFF);
        size_buf[1] = (uint8_t)((frame_size >> 8) & 0xFF);
        size_buf[2] = (uint8_t)((frame_size >> 16) & 0xFF);
        size_buf[3] = (uint8_t)((frame_size >> 24) & 0xFF);

        total_sent = 0;
        while (total_sent < sizeof(size_buf)) {
            n = ::send(m_sock, size_buf + total_sent, sizeof(size_buf) - total_sent, 0);
            if (n <= 0) break;
            total_sent += n;
        }

        total_sent = 0;
        while (total_sent < jpegs[fi].size()) {
            n = ::send(m_sock, jpegs[fi].data() + total_sent,
                       jpegs[fi].size() - total_sent, 0);
            if (n <= 0) break;
            total_sent += n;
        }
    }

    std::cout << "[SEND] Goal #" << seq << " team=" << team_id
              << " frames=" << jpegs.size() << std::endl;

    ::close(m_sock);
    m_sock = -1;
}

void GoalSender::close() {
    if (m_sock >= 0) {
        ::close(m_sock);
        m_sock = -1;
    }
}
