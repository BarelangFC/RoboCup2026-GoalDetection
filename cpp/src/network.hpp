#ifndef NETWORK_H
#define NETWORK_H

#include <string>

class UdpDispatcher {
public:
    UdpDispatcher();
    ~UdpDispatcher();

    bool setup(const std::string& target_ip, int target_port, bool broadcast);
    void send_goal_event(int seq_num, int team_id, float x, float y);
    void close();

private:
    int m_sock = -1;
    std::string m_target_ip;
    int m_target_port = 5000;

    struct GoalPacket {
        uint16_t magic = 0x474F;
        uint16_t seq;
        uint32_t timestamp;
        uint8_t event_type = 1;
        uint8_t team_id = 1;
        uint8_t reserved[6] = {0};
    };
};

#endif
