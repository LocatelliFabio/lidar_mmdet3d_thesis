#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include <rs_driver/api/lidar_driver.hpp>
#include <rs_driver/driver/driver_param.hpp>

namespace py = pybind11;
using namespace robosense::lidar;

// -------------------------
// Point type expected by rs_driver decoders (must have x,y,z,intensity)
// -------------------------
struct PointXYZI
{
  float x = 0.f, y = 0.f, z = 0.f;
  uint8_t intensity = 0;
};

// -------------------------
// PointCloud type expected by rs_driver templates
// MUST provide: PointT typedef, points container, frame_id, timestamp, seq, width/height/is_dense
// -------------------------
template <typename T_Point>
struct PointCloudT
{
  using PointT = T_Point;                 // <-- FIX CRITICO
  std::vector<PointT> points;

  uint32_t height = 1;
  uint32_t width = 0;
  bool is_dense = false;

  std::string frame_id;                   // <-- rs_driver uses this
  double timestamp = 0.0;
  uint32_t seq = 0;
};

using PointCloudMsg = PointCloudT<PointXYZI>;

// -------------------------
// Drop-old queue (keeps latency low)
// -------------------------
template <typename T>
class DropOldQueue
{
public:
  explicit DropOldQueue(size_t max_size = 2) : max_size_(max_size ? max_size : 1) {}

  void set_max_size(size_t n)
  {
    std::lock_guard<std::mutex> lk(m_);
    max_size_ = (n ? n : 1);
    while (q_.size() > max_size_) q_.pop_front();
  }

  void push(T v)
  {
    std::lock_guard<std::mutex> lk(m_);
    if (q_.size() >= max_size_) q_.pop_front();
    q_.push_back(std::move(v));
    cv_.notify_one();
  }

  bool pop(T& out, int timeout_ms)
  {
    std::unique_lock<std::mutex> lk(m_);
    if (timeout_ms < 0) {
      cv_.wait(lk, [&]{ return !q_.empty(); });
    } else if (timeout_ms == 0) {
      if (q_.empty()) return false;
    } else {
      auto ok = cv_.wait_for(lk, std::chrono::milliseconds(timeout_ms), [&]{ return !q_.empty(); });
      if (!ok) return false;
    }
    out = std::move(q_.front());
    q_.pop_front();
    return true;
  }

  void clear()
  {
    std::lock_guard<std::mutex> lk(m_);
    q_.clear();
  }

private:
  std::mutex m_;
  std::condition_variable cv_;
  std::deque<T> q_;
  size_t max_size_;
};

// -------------------------
// Bridge
// -------------------------
class RsBridge
{
public:
  RsBridge() : stuffed_q_(2) {}
  ~RsBridge() { stop(); }

  // use_rsp128: True => LidarType::RSP128, False => LidarType::RS128
  void configure(const std::string& host_ip,
                 uint16_t msop_port,
                 uint16_t difop_port,
                 bool use_rsp128 = true,
                 float split_angle_deg = 0.0f,
                 int queue_size = 2,
                 int free_pool_size = 16,
                 size_t prealloc_points = 250000)
  {
    if (running_) throw std::runtime_error("Cannot configure while running.");

    host_ip_ = host_ip;
    msop_port_ = msop_port;
    difop_port_ = difop_port;
    use_rsp128_ = use_rsp128;

    split_angle_deg_ = split_angle_deg;
    stuffed_q_.set_max_size((size_t)(queue_size <= 0 ? 1 : queue_size));

    free_pool_size_ = (free_pool_size <= 0 ? 1 : free_pool_size);
    prealloc_points_ = prealloc_points;

    // init free pool
    {
      std::lock_guard<std::mutex> lk(pool_mtx_);
      pool_.clear();
      pool_.reserve((size_t)free_pool_size_);
      for (int i = 0; i < free_pool_size_; ++i) {
        auto msg = std::make_shared<PointCloudMsg>();
        msg->points.reserve(prealloc_points_);
        pool_.push_back(std::move(msg));
      }
    }

    configured_ = true;
  }

  void start()
  {
    if (!configured_) throw std::runtime_error("Call configure() before start().");
    if (running_) return;

    RSDriverParam param;
    param.input_type = InputType::ONLINE_LIDAR;
    param.input_param.host_address = host_ip_;
    param.input_param.msop_port = msop_port_;
    param.input_param.difop_port = difop_port_;

    // full revolution frames
    param.decoder_param.split_frame_mode = SplitFrameMode::SPLIT_BY_ANGLE;
    param.decoder_param.split_angle = split_angle_deg_;

    // IMPORTANT: your driver supports RSP128 decoders (you showed decoder_RSP128.hpp),
    // so default is RSP128
    param.lidar_type = use_rsp128_ ? LidarType::RSP128 : LidarType::RS128;

    driver_.regPointCloudCallback(
      std::bind(&RsBridge::cb_get_cloud, this),
      std::bind(&RsBridge::cb_put_cloud, this, std::placeholders::_1)
    );

    if (!driver_.init(param)) {
      throw std::runtime_error("rs_driver init() failed (check ports/network/lidar_type).");
    }

    running_ = true;
    driver_.start();
  }

  void stop()
  {
    if (!running_) return;
    running_ = false;

    try { driver_.stop(); } catch (...) {}
    stuffed_q_.clear();
  }

  bool is_running() const { return running_; }

  // Returns numpy float32 (N,4) or None
  py::object get_frame(int timeout_ms = 0)
  {
    std::shared_ptr<PointCloudMsg> msg;
    if (!stuffed_q_.pop(msg, timeout_ms)) return py::none();
    if (!msg) return py::none();

    const size_t n = msg->points.size();

    auto buf = std::make_shared<std::vector<float>>();
    buf->resize(n * 4);

    float* out = buf->data();
    for (size_t i = 0; i < n; ++i) {
      const auto& p = msg->points[i];
      out[i * 4 + 0] = p.x;
      out[i * 4 + 1] = p.y;
      out[i * 4 + 2] = p.z;
      out[i * 4 + 3] = static_cast<float>(p.intensity);
    }

    recycle_cloud(std::move(msg));

    auto capsule = py::capsule(new std::shared_ptr<std::vector<float>>(buf),
                               [](void* p) {
                                 delete reinterpret_cast<std::shared_ptr<std::vector<float>>*>(p);
                               });

    py::ssize_t N = (py::ssize_t)n;
    return py::array_t<float>(
      {N, (py::ssize_t)4},
      {(py::ssize_t)(4 * sizeof(float)), (py::ssize_t)sizeof(float)},
      buf->data(),
      capsule
    );
  }

private:
  std::shared_ptr<PointCloudMsg> cb_get_cloud()
  {
    std::lock_guard<std::mutex> lk(pool_mtx_);
    if (!pool_.empty()) {
      auto msg = pool_.back();
      pool_.pop_back();

      // reset but keep capacity
      msg->height = 1;
      msg->width = 0;
      msg->is_dense = false;
      msg->timestamp = 0.0;
      msg->seq = 0;
      msg->frame_id.clear();
      msg->points.clear();
      return msg;
    }

    // fallback allocation
    auto msg = std::make_shared<PointCloudMsg>();
    msg->points.reserve(prealloc_points_);
    return msg;
  }

  void cb_put_cloud(std::shared_ptr<PointCloudMsg> msg)
  {
    if (!msg) return;
    if (!running_) { recycle_cloud(std::move(msg)); return; }
    stuffed_q_.push(std::move(msg));
  }

  void recycle_cloud(std::shared_ptr<PointCloudMsg> msg)
  {
    if (!msg) return;
    std::lock_guard<std::mutex> lk(pool_mtx_);
    if ((int)pool_.size() < free_pool_size_) pool_.push_back(std::move(msg));
  }

private:
  LidarDriver<PointCloudMsg> driver_;

  std::atomic<bool> running_{false};
  bool configured_ = false;

  std::string host_ip_ = "0.0.0.0";
  uint16_t msop_port_ = 6699;
  uint16_t difop_port_ = 7788;

  bool use_rsp128_ = true;
  float split_angle_deg_ = 0.0f;

  DropOldQueue<std::shared_ptr<PointCloudMsg>> stuffed_q_;

  std::mutex pool_mtx_;
  std::vector<std::shared_ptr<PointCloudMsg>> pool_;
  int free_pool_size_ = 16;
  size_t prealloc_points_ = 250000;
};

// -------------------------
// pybind11 module
// -------------------------
PYBIND11_MODULE(rs_bridge, m)
{
  py::class_<RsBridge>(m, "RsBridge")
    .def(py::init<>())
    .def("configure", &RsBridge::configure,
         py::arg("host_ip") = "0.0.0.0",
         py::arg("msop_port") = 6699,
         py::arg("difop_port") = 7788,
         py::arg("use_rsp128") = true,
         py::arg("split_angle_deg") = 0.0f,
         py::arg("queue_size") = 2,
         py::arg("free_pool_size") = 16,
         py::arg("prealloc_points") = (size_t)250000)
    .def("start", &RsBridge::start)
    .def("stop", &RsBridge::stop)
    .def("is_running", &RsBridge::is_running)
    .def("get_frame", &RsBridge::get_frame, py::arg("timeout_ms") = 0);
}