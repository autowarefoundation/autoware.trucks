#include <visualization/occupancy_bridge.hpp>
#include <visualization/visual_interface.hpp>

#include <cmath>
#include <vector>

namespace visualization
{
namespace occupancy
{

namespace
{

constexpr int kNetW = 1024;
constexpr int kNetH = 512;

}  // namespace

Scene make_scene(
  const visionpilot::models::InferenceFrameResult & r, const Plan & /*plan*/,
  const cv::Mat & H_resized)
{
  Scene s;

  if (!H_resized.empty()) {
    cv::Mat H64;
    H_resized.convertTo(H64, CV_64F);
    H64.convertTo(s.H_px2world, CV_32F);
  }

  if (r.lateral.path_valid) {
    s.path_a = r.lateral.path_a;
    s.path_b = r.lateral.path_b;
    s.path_c = r.lateral.path_c;
    s.path_valid = true;
  }
  // Always take filtered CTE/yaw so ego can animate during lane changes
  // even when the RANSAC path briefly drops out.
  if (r.lateral.valid) {
    s.cte_m = r.lateral.cte_m;
    s.yaw_rad = r.lateral.yaw_rad;
  } else if (r.lateral.path_valid) {
    s.cte_m = r.lateral.raw_cte_m;
    s.yaw_rad = r.lateral.raw_yaw_rad;
  }

  if (r.auto_steer.valid && !s.H_px2world.empty()) {
    constexpr int kSteerN = 64;
    s.lane_world.reserve(kSteerN);
    std::vector<cv::Point2f> src;
    src.reserve(kSteerN);
    for (int i = 0; i < kSteerN; ++i) {
      if (r.auto_steer.h_vector[static_cast<size_t>(i)] < 0.5f) continue;
      const float u = r.auto_steer.xp[static_cast<size_t>(i)] * static_cast<float>(kNetW);
      const float v =
        static_cast<float>(i) * (static_cast<float>(kNetH - 1) / static_cast<float>(kSteerN - 1));
      src.emplace_back(u, v);
    }
    if (!src.empty()) {
      std::vector<cv::Point2f> dst;
      cv::perspectiveTransform(src, dst, s.H_px2world);
      for (const auto & p : dst) {
        if (!std::isfinite(p.x) || !std::isfinite(p.y)) continue;
        if (p.x < -1.f || p.x > 80.f) continue;
        s.lane_world.push_back(p);
      }
    }
  }

  s.detections.reserve(r.auto_speed.detections.size());
  for (const auto & d : r.auto_speed.detections) {
    s.detections.push_back({d.x1, d.y1, d.x2, d.y2, d.score, d.class_id});
  }

  s.cipo_valid = r.cipo.valid;
  s.cipo_distance_m = r.cipo.distance_m;

  // AutoDrive-only CIPO: AD confirms in-path object, AutoSpeed has no bbox.
  static constexpr float kDMaxM = 150.f;
  if (r.auto_drive.valid && r.auto_drive.flag_prob >= 0.40f && !r.cipo.cipo_raw_found) {
    s.ad_cipo_only = true;
    s.ad_distance_m = kDMaxM * (1.f - r.auto_drive.dist_normalized);
  }

  return s;
}

cv::Mat build_frame(
  const visionpilot::models::InferenceFrameResult & result, const Plan & plan,
  const cv::Mat & H_resized)
{
  return render(make_scene(result, plan, H_resized));
}

void publish(
  VisualInterface * visual_interface, const visionpilot::models::InferenceFrameResult & result,
  const Plan & plan, const cv::Mat & H_resized)
{
  if (!visual_interface) return;
  visual_interface->set_aux_frame(build_frame(result, plan, H_resized));
}

}  // namespace occupancy
}  // namespace visualization
