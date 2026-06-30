#ifndef GOAL_SENDER_H
#define GOAL_SENDER_H

#include <string>
#include <vector>
#include <cstdint>

/// Sends goal footage to the GameController via TCP.
/// Packet format:
///   [team_id:1][seq:2][timestamp:4][num_frames:1]
///   then for each frame: [jpeg_size:4][jpeg_data...]
class GoalSender {
public:
    GoalSender();
    ~GoalSender();

    bool setup(const std::string& host, int port);
    void send(int team_id, int seq, const std::vector<std::vector<unsigned char>>& jpegs);
    void close();
    bool is_connected() const { return m_sock >= 0; }

private:
    int m_sock = -1;
    std::string m_host;
    int m_port = 3737;
};

#endif
