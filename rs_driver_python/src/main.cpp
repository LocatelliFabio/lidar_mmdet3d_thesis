#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <rs_driver/api/lidar_driver.hpp>
#include <rs_driver/driver/driver_param.hpp>
#include <rs_driver/msg/point_cloud_msg.hpp>

#define STRINGIFY(x) #x

namespace py = pybind11;
namespace rsld = robosense::lidar;

typedef PointCloudT<PointXYZI> PointCloud_XYZI;
typedef rsld::LidarDriver<PointCloud_XYZI> LidarDriver_XYZI;

py::array_t<float> points_to_numpy_xyzi(const PointCloud_XYZI &point_cloud)
{
    constexpr size_t dims = 2;
    size_t num_rows = point_cloud.width * point_cloud.height;
    constexpr size_t num_columns = 4;
    size_t shape[dims]{num_rows, num_columns};

    auto data_array = py::array_t<float>(shape);
    auto unchecked_ref = data_array.mutable_unchecked<dims>();

    for (size_t row = 0; row < num_rows; row++)
    {
        unchecked_ref(row, 0) = -point_cloud.points[row].y;
        unchecked_ref(row, 1) = -point_cloud.points[row].z;
        unchecked_ref(row, 2) = point_cloud.points[row].x;
        unchecked_ref(row, 3) = static_cast<float>(point_cloud.points[row].intensity);
    }

    return data_array;
}

PYBIND11_MODULE(rs_driver, m)
{
    /* ERRORS */

    py::enum_<rsld::ErrCodeType>(m, "ErrCodeType")
        .value("INFO_CODE", rsld::ErrCodeType::INFO_CODE)
        .value("WARNING_CODE", rsld::ErrCodeType::WARNING_CODE)
        .value("ERROR_CODE", rsld::ErrCodeType::ERROR_CODE);

    py::enum_<rsld::ErrCode>(m, "ErrCode")
        .value("ERRCODE_SUCCESS", rsld::ErrCode::ERRCODE_SUCCESS)
        .value("ERRCODE_PCAPREPEAT", rsld::ErrCode::ERRCODE_PCAPREPEAT)
        .value("ERRCODE_PCAPEXIT", rsld::ErrCode::ERRCODE_PCAPEXIT)
        .value("ERRCODE_MSOPTIMEOUT", rsld::ErrCode::ERRCODE_MSOPTIMEOUT)
        .value("ERRCODE_NODIFOPRECV", rsld::ErrCode::ERRCODE_NODIFOPRECV)
        .value("ERRCODE_WRONGMSOPLEN", rsld::ErrCode::ERRCODE_WRONGMSOPLEN)
        .value("ERRCODE_WRONGMSOPID", rsld::ErrCode::ERRCODE_WRONGMSOPID)
        .value("ERRCODE_WRONGMSOPBLKID", rsld::ErrCode::ERRCODE_WRONGMSOPBLKID)
        .value("ERRCODE_WRONGDIFOPLEN", rsld::ErrCode::ERRCODE_WRONGDIFOPLEN)
        .value("ERRCODE_WRONGDIFOPID", rsld::ErrCode::ERRCODE_WRONGDIFOPID)
        .value("ERRCODE_ZEROPOINTS", rsld::ErrCode::ERRCODE_ZEROPOINTS)
        .value("ERRCODE_PKTBUFOVERFLOW", rsld::ErrCode::ERRCODE_PKTBUFOVERFLOW)
        .value("ERRCODE_CLOUDOVERFLOW", rsld::ErrCode::ERRCODE_CLOUDOVERFLOW)
        .value("ERRCODE_WRONGCRC32", rsld::ErrCode::ERRCODE_WRONGCRC32)
        .value("ERRCODE_STARTBEFOREINIT", rsld::ErrCode::ERRCODE_STARTBEFOREINIT)
        .value("ERRCODE_PCAPWRONGPATH", rsld::ErrCode::ERRCODE_PCAPWRONGPATH)
        .value("ERRCODE_POINTCLOUDNULL", rsld::ErrCode::ERRCODE_POINTCLOUDNULL);

    py::class_<rsld::Error>(m, "Error")
        .def_readwrite("error_code", &rsld::Error::error_code)
        .def_readwrite("error_code_type", &rsld::Error::error_code_type)
        .def("to_string", &rsld::Error::toString)
        .def("__str__", &rsld::Error::toString);

    /* DRIVER PARAMETERS */

    py::enum_<rsld::InputType>(m, "InputType")
        .value("ONLINE_LIDAR", rsld::InputType::ONLINE_LIDAR)
        .value("PCAP_FILE", rsld::InputType::PCAP_FILE)
        .value("RAW_PACKET", rsld::InputType::RAW_PACKET);

    py::enum_<rsld::LidarType>(m, "LidarType")
        .value("RS_MECH", rsld::LidarType::RS_MECH)
        .value("RS16", rsld::LidarType::RS16)
        .value("RS32", rsld::LidarType::RS32)
        .value("RSBP", rsld::LidarType::RSBP)
        .value("RSHELIOS", rsld::LidarType::RSHELIOS)
        .value("RSHELIOS_16P", rsld::LidarType::RSHELIOS_16P)
        .value("RS128", rsld::LidarType::RS128)
        .value("RS80", rsld::LidarType::RS80)
        .value("RS48", rsld::LidarType::RS48)
        .value("RSP128", rsld::LidarType::RSP128)
        .value("RSP80", rsld::LidarType::RSP80)
        .value("RSP48", rsld::LidarType::RSP48)
        .value("RS_MEMS", rsld::LidarType::RS_MEMS)
        .value("RSM1", rsld::LidarType::RSM1)
        .value("RSM2", rsld::LidarType::RSM2)
        .value("RSM3", rsld::LidarType::RSM3)
        .value("RSE1", rsld::LidarType::RSE1)
        .value("RSMX", rsld::LidarType::RSMX)
        .value("RS_JUMBO", rsld::LidarType::RS_JUMBO)
        .value("RSM1_JUMBO", rsld::LidarType::RSM1_JUMBO);

    py::class_<rsld::RSInputParam>(m, "RSInputParam")
        .def(py::init<>())
        .def_readwrite("msop_port", &rsld::RSInputParam::msop_port)
        .def_readwrite("difop_port", &rsld::RSInputParam::difop_port)
        .def_readwrite("imu_port", &rsld::RSInputParam::imu_port)
        .def_readwrite("user_layer_bytes", &rsld::RSInputParam::user_layer_bytes)
        .def_readwrite("tail_layer_bytes", &rsld::RSInputParam::tail_layer_bytes)
        .def_readwrite("host_address", &rsld::RSInputParam::host_address)
        .def_readwrite("group_address", &rsld::RSInputParam::group_address)
        .def_readwrite("socket_recv_buf", &rsld::RSInputParam::socket_recv_buf)
        .def_readwrite("pcap_path", &rsld::RSInputParam::pcap_path)
        .def_readwrite("pcap_rate", &rsld::RSInputParam::pcap_rate)
        .def_readwrite("pcap_repeat", &rsld::RSInputParam::pcap_repeat)
        .def_readwrite("use_vlan", &rsld::RSInputParam::use_vlan)
        .def("print", &rsld::RSInputParam::print);

    py::class_<rsld::RSDecoderParam>(m, "RSDecoderParam")
        .def(py::init<>())
        .def_readwrite("min_distance", &rsld::RSDecoderParam::min_distance)
        .def_readwrite("max_distance", &rsld::RSDecoderParam::max_distance)
        .def_readwrite("use_lidar_clock", &rsld::RSDecoderParam::use_lidar_clock)
        .def_readwrite("dense_points", &rsld::RSDecoderParam::dense_points)
        .def_readwrite("ts_first_point", &rsld::RSDecoderParam::ts_first_point)
        .def_readwrite("wait_for_difop", &rsld::RSDecoderParam::wait_for_difop)
        .def_readwrite("config_from_file", &rsld::RSDecoderParam::config_from_file)
        .def_readwrite("angle_path", &rsld::RSDecoderParam::angle_path)
        .def_readwrite("start_angle", &rsld::RSDecoderParam::start_angle)
        .def_readwrite("end_angle", &rsld::RSDecoderParam::end_angle)
        .def_readwrite("split_frame_mode", &rsld::RSDecoderParam::split_frame_mode)
        .def_readwrite("split_angle", &rsld::RSDecoderParam::split_angle)
        .def_readwrite("num_blks_split", &rsld::RSDecoderParam::num_blks_split);

    py::class_<rsld::RSDriverParam>(m, "RSDriverParam")
        .def(py::init<>())
        .def_readwrite("lidar_type", &rsld::RSDriverParam::lidar_type)
        .def_readwrite("input_type", &rsld::RSDriverParam::input_type)
        .def_readwrite("frame_id", &rsld::RSDriverParam::frame_id)
        .def_readwrite("input_param", &rsld::RSDriverParam::input_param)
        .def_readwrite("decoder_param", &rsld::RSDriverParam::decoder_param)
        .def("print", &rsld::RSDriverParam::print);

    /* POINT CLOUD TYPES */

    py::class_<PointXYZI>(m, "PointXYZI")
        .def(py::init<>())
        .def_readwrite("x", &PointXYZI::x)
        .def_readwrite("y", &PointXYZI::y)
        .def_readwrite("z", &PointXYZI::z)
        .def_readwrite("intensity", &PointXYZI::intensity);

    py::class_<PointCloud_XYZI, std::shared_ptr<PointCloud_XYZI>>(m, "PointCloud")
        .def(py::init<>())
        .def_readwrite("height", &PointCloud_XYZI::height)
        .def_readwrite("width", &PointCloud_XYZI::width)
        .def_readwrite("is_dense", &PointCloud_XYZI::is_dense)
        .def_readwrite("timestamp", &PointCloud_XYZI::timestamp)
        .def_readwrite("seq", &PointCloud_XYZI::seq)
        .def_readwrite("frame_id", &PointCloud_XYZI::frame_id)
        .def("__str__", [](const PointCloud_XYZI &a)
        {
            return "Point cloud " + std::to_string(a.seq) +
                   " of size " + std::to_string(a.points.size());
        })
        .def("numpy", &points_to_numpy_xyzi);

    /* LIDAR DRIVER */

    py::class_<LidarDriver_XYZI>(m, "LidarDriver")
        .def(py::init<>())
        .def("init", &LidarDriver_XYZI::init)
        .def("register_point_cloud_callback", &LidarDriver_XYZI::regPointCloudCallback)
        .def("register_exception_callback", &LidarDriver_XYZI::regExceptionCallback)
        .def("start", &LidarDriver_XYZI::start)
        .def("stop", &LidarDriver_XYZI::stop);

#ifdef VERSION_INFO
    m.attr("__version__") = STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
