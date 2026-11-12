#ifndef VISUALIZATION__VISUAL_INTERFACE_HPP_
#define VISUALIZATION__VISUAL_INTERFACE_HPP_
#include <opencv2/highgui.hpp>

class VisualInterface
{
public:
  VisualInterface();
  virtual ~VisualInterface() = default;

  virtual bool render_frame(const cv::Mat & display_frame) = 0;
  // Optional second view (e.g. occupancy BEV). Default: ignore.
  virtual void set_aux_frame(const cv::Mat & /*aux*/) {}
  virtual bool stop() = 0;
};

#endif  // VISUALIZATION__VISUAL_INTERFACE_HPP_
