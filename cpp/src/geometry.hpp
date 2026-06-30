#ifndef GEOMETRY_H
#define GEOMETRY_H

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "yolo.hpp"

enum GoalMode { GOAL_CENTER, GOAL_FULL, GOAL_OVERLAP_PCT };
enum TeamId { TEAM_NONE = 0, TEAM_1 = 1, TEAM_2 = 2 };

struct GoalZone {
    std::vector<cv::Point> polygon;
    TeamId team = TEAM_NONE;
    int goal_count = 0;
    double last_goal_time = 0;
    bool ready() const { return polygon.size() >= 3; }
};

class GoalChecker {
public:
    void set_polygon(TeamId team, const std::vector<cv::Point>& poly);
    bool has_polygon(TeamId team) const;
    void set_mode(GoalMode m) { m_mode = m; }
    void set_overlap_pct(float p) { m_overlap_pct = p; }
    void reset_counts();
    int get_goal_count(TeamId team) const;

    // Check all goal zones. Returns which team scored (TEAM_NONE = no goal)
    TeamId check_goals(const Detection& det, float& goal_x, float& goal_y);

    const std::vector<cv::Point>& get_polygon(TeamId team) const;

private:
    GoalZone m_team1, m_team2;
    double m_cooldown = 2.0;
    GoalMode m_mode = GOAL_FULL;
    float m_overlap_pct = 0.5f;

    bool point_in_poly(const std::vector<cv::Point>& poly, float px, float py) const;
    float bbox_overlap(const std::vector<cv::Point>& poly, const Detection& det) const;
    bool check_zone(GoalZone& zone, const Detection& det, float& gx, float& gy);
};

#endif
