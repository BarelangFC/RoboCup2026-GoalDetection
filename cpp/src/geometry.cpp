#include "geometry.hpp"
#include <chrono>
#include <cmath>

void GoalChecker::set_polygon(TeamId team, const std::vector<cv::Point>& poly) {
    if (team == TEAM_1) { m_team1.polygon = poly; m_team1.team = TEAM_1; }
    if (team == TEAM_2) { m_team2.polygon = poly; m_team2.team = TEAM_2; }
}

bool GoalChecker::has_polygon(TeamId team) const {
    return team == TEAM_1 ? m_team1.ready() : m_team2.ready();
}

const std::vector<cv::Point>& GoalChecker::get_polygon(TeamId team) const {
    return team == TEAM_1 ? m_team1.polygon : m_team2.polygon;
}

int GoalChecker::get_goal_count(TeamId team) const {
    return team == TEAM_1 ? m_team1.goal_count : m_team2.goal_count;
}

void GoalChecker::reset_counts() {
    m_team1.goal_count = 0;
    m_team2.goal_count = 0;
}

bool GoalChecker::point_in_poly(const std::vector<cv::Point>& poly, float px, float py) const {
    if (poly.empty()) return false;
    return cv::pointPolygonTest(poly, cv::Point2f(px, py), false) >= 0;
}

float GoalChecker::bbox_overlap(const std::vector<cv::Point>& poly, const Detection& det) const {
    if (poly.empty()) return 0.0f;
    float bx1 = det.x1, by1 = det.y1, bx2 = det.x2, by2 = det.y2;
    if (bx2 <= bx1 || by2 <= by1) return 0.0f;
    int steps_x = std::min(10, std::max(2, (int)(bx2 - bx1) / 4));
    int steps_y = std::min(10, std::max(2, (int)(by2 - by1) / 4));
    float step_x = (bx2 - bx1) / steps_x;
    float step_y = (by2 - by1) / steps_y;
    int inside = 0, total = 0;
    for (int yi = 0; yi < steps_y; yi++) {
        for (int xi = 0; xi < steps_x; xi++) {
            float px = bx1 + (xi + 0.5f) * step_x;
            float py = by1 + (yi + 0.5f) * step_y;
            total++;
            if (point_in_poly(poly, px, py)) inside++;
        }
    }
    return total > 0 ? (float)inside / total : 0.0f;
}

bool GoalChecker::check_zone(GoalZone& zone, const Detection& det, float& gx, float& gy) {
    if (!zone.ready()) return false;
    bool result = false;
    float bx = (det.x1 + det.x2) / 2.0f;
    float by = (det.y1 + det.y2) / 2.0f;
    gx = bx; gy = by;

    if (m_mode == GOAL_CENTER) {
        result = point_in_poly(zone.polygon, bx, det.y2);
    } else if (m_mode == GOAL_FULL) {
        result = point_in_poly(zone.polygon, det.x1, det.y1) &&
                 point_in_poly(zone.polygon, det.x2, det.y1) &&
                 point_in_poly(zone.polygon, det.x2, det.y2) &&
                 point_in_poly(zone.polygon, det.x1, det.y2);
    } else if (m_mode == GOAL_OVERLAP_PCT) {
        result = bbox_overlap(zone.polygon, det) >= m_overlap_pct;
    }

    if (result) {
        double now = std::chrono::duration<double>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        if (now - zone.last_goal_time > m_cooldown) {
            zone.last_goal_time = now;
            zone.goal_count++;
            return true;
        }
    }
    return false;
}

TeamId GoalChecker::check_goals(const Detection& det, float& goal_x, float& goal_y) {
    if (check_zone(m_team1, det, goal_x, goal_y)) return TEAM_1;
    if (check_zone(m_team2, det, goal_x, goal_y)) return TEAM_2;
    return TEAM_NONE;
}
