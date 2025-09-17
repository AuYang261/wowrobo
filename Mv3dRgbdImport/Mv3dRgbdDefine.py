# -- coding: utf-8 --
from ctypes import *

STRING = c_char_p
int8_t = c_int8
int16_t = c_int16
int32_t = c_int32
int64_t = c_int64
uint8_t = c_uint8
uint16_t = c_uint16
uint32_t = c_uint32
uint64_t = c_uint64
int_least8_t = c_byte
int_least16_t = c_short
int_least32_t = c_int
int_least64_t = c_long
uint_least8_t = c_ubyte
uint_least16_t = c_ushort
uint_least32_t = c_uint
uint_least64_t = c_ulong
int_fast8_t = c_byte
int_fast16_t = c_long
int_fast32_t = c_long
int_fast64_t = c_long
uint_fast8_t = c_ubyte
uint_fast16_t = c_ulong
uint_fast32_t = c_ulong
uint_fast64_t = c_ulong
intptr_t = c_long
uintptr_t = c_ulong
intmax_t = c_long
uintmax_t = c_ulong

MV3D_RGBD_UNDEFINED                     = -1
# ch: 状态码 | en: Status Code 
# ch: 正确码定义 | en: Definition of correct code
MV3D_RGBD_OK                            = 0                                    #~chinese 成功，无错误                  ~english Successed, no error

# ch: 通用错误码定义:范围0x80060000-0x800600FF | en: Definition of General Error Code (from 0x80060000 to 0x800600FF)
MV3D_RGBD_E_HANDLE                      = 0x80060000                           #~chinese 错误或无效的句柄              ~english Incorrect or invalid handle  
MV3D_RGBD_E_SUPPORT                     = 0x80060001                           #~chinese 不支持的功能                  ~english The function is not supported 
MV3D_RGBD_E_BUFOVER                     = 0x80060002                           #~chinese 缓存已满                      ~english The buffer is full
MV3D_RGBD_E_CALLORDER                   = 0x80060003                           #~chinese 函数调用顺序错误              ~english Incorrect calling sequence
MV3D_RGBD_E_PARAMETER                   = 0x80060004                           #~chinese 错误的参数                    ~english Incorrect parameter
MV3D_RGBD_E_RESOURCE                    = 0x80060005                           #~chinese 资源申请失败                  english Resource request failed
MV3D_RGBD_E_NODATA                      = 0x80060006                           #~chinese 无数据                        ~english No data
MV3D_RGBD_E_PRECONDITION                = 0x80060007                           #~chinese 前置条件有误，或运行环境已发生变化      ~english Incorrect precondition, or running environment has changed 
MV3D_RGBD_E_VERSION                     = 0x80060008                           #~chinese 版本不匹配                    ~english The version mismatched
MV3D_RGBD_E_NOENOUGH_BUF                = 0x80060009                           #~chinese 传入的内存空间不足            ~english Insufficient memory
MV3D_RGBD_E_ABNORMAL_IMAGE              = 0x8006000A                           #~chinese 异常图像，可能是丢包导致图像不完整      ~english Abnormal image. Incomplete image caused by packet loss 
MV3D_RGBD_E_LOAD_LIBRARY                = 0x8006000B                           #~chinese 动态导入DLL失败               ~english Failed to load the dynamic link library dynamically
MV3D_RGBD_E_ALGORITHM                   = 0x8006000C                           #~chinese 算法错误                      ~english Algorithm error
MV3D_RGBD_E_DEVICE_OFFLINE              = 0x8006000D                           #~chinese 设备离线                      ~english The device is offline 
MV3D_RGBD_E_ACCESS_DENIED               = 0x8006000E                           #~chinese 设备无访问权限                ~english No device access permission
MV3D_RGBD_E_OUTOFRANGE                  = 0x8006000F                           #~chinese 值超出范围                    ~english The value is out of range

MV3D_RGBD_E_UPG_FILE_MISMATCH           = 0x80060010                           #~chinese 升级固件不匹配                ~english The upgraded firmware does not match
MV3D_RGBD_E_UPG_CONFLICT                = 0x80060012                           #~chinese 升级冲突                      ~english The upgraded firmware conflict
MV3D_RGBD_E_UPG_INNER_ERR               = 0x80060013                           #~chinese 升级时相机内部出现错误        ~english An error occurred inside the camera during the upgrade 

MV3D_RGBD_E_INDUSTRY                    = 0x80060020                           #~chinese 行业属性不匹配                ~english The industry attributes does not match
MV3D_RGBD_E_NETWORK                     = 0x80060021                           #~chinese 网络相关错误                  ~english Network related error

MV3D_RGBD_E_USB_WRITE                   = 0x80060030                           #~chinese 写USB出错                     ~english Writing USB error

MV3D_RGBD_E_UNKNOW                      = 0x800600FF                           #~chinese 未知的错误                    ~english Unknown error

# ch: 常量定义 | en: Macro Definition
MV3D_RGBD_MAX_IMAGE_COUNT               = 10                                   #~chinese 最大图片个数                  ~english The maximum number of images
MV3D_RGBD_MAX_STRING_LENGTH             = 256                                  #~chinese 最大字符串长度                ~english The maximum length of string
MV3D_RGBD_MAX_PATH_LENGTH               = 256                                  #~chinese 最大路径长度                  ~english The maximum length of path
MV3D_RGBD_MAX_ENUM_COUNT                = 16                                   #~chinese 最大枚举数量                  ~english The maximum number of enumerations

# ch: 像素类型 | en: Pixel Type
MV3D_RGBD_PIXEL_MONO                    = 0x01000000                           #~chinese Mono像素格式                  ~english Mono pixel format
MV3D_RGBD_PIXEL_COLOR                   = 0x02000000                           #~chinese Color像素格式                 ~english Color pixel format
MV3D_RGBD_PIXEL_CUSTOM                  = 0x80000000                           #~chinese 自定义像素格式                ~english Custom pixel format

# ch:设备属性key值相关定义 | en: Attribute Key Value Definition
MV3D_RGBD_INT_WIDTH                     = "Width"                              #~chinese 图像宽                        ~english Image width
MV3D_RGBD_INT_HEIGHT                    = "Height"                             #~chinese 图像高                        ~english Image height
MV3D_RGBD_ENUM_WORKINGMODE              = "CameraWorkingMode"                  #~chinese 工作模式                      ~english The camera working mode
MV3D_RGBD_ENUM_PIXELFORMAT              = "PixelFormat"                        #~chinese 像素格式                      ~english Pixel format
MV3D_RGBD_ENUM_IMAGEMODE                = "ImageMode"                          #~chinese 图像模式                      ~english Image mode
MV3D_RGBD_FLOAT_GAIN                    = "Gain"                               #~chinese 增益                          ~english Gain
MV3D_RGBD_FLOAT_EXPOSURETIME            = "ExposureTime"                       #~chinese 曝光时间                      ~english Exposure time
MV3D_RGBD_FLOAT_FRAMERATE               = "AcquisitionFrameRate"               #~chinese 采集帧率                      ~english Acquired frame rate
MV3D_RGBD_ENUM_TRIGGERSELECTOR          = "TriggerSelector"                    #~chinese 触发选择器                    ~english Trigger selector
MV3D_RGBD_ENUM_TRIGGERMODE              = "TriggerMode"                        #~chinese 触发模式                      ~english Trigger mode
MV3D_RGBD_ENUM_TRIGGERSOURCE            = "TriggerSource"                      #~chinese 触发源                        ~english Trigger source 
MV3D_RGBD_FLOAT_TRIGGERDELAY            = "TriggerDelay"                       #~chinese 触发延迟时间                  ~english Trigger delay
MV3D_RGBD_INT_IMAGEALIGN                = "ImageAlign"                         #~chinese 对齐坐标系（默认值1：对齐到彩色图坐标系；0：不对齐；2：对齐到深度图坐标系），重启程序后恢复默认值 
                                                                               #~english Whether to align the depth image with RGB image: 1 (align, default value), 0 (not align). The default value will be restored after rebooting
MV3D_RGBD_BOOL_LASERENABLE              = "LaserEnable"                        #~chinese 投射器使能                    ~english Open or close laser control
Mv3D_RGBD_FLOAT_BASEDISTANCE            = "BaseDistance"                       #~chinese 左右目基线距                  ~english Left and right eyes base distance
MV3D_RGBD_ENUM_RESOLUTION               = "BinningSelector"                    #~chinese 采集分辨率                    ~english Acquisition resolution
MV3D_RGBD_INT_OUTPUT_RGBD               = "OutputRgbd"                         #~chinese RGBD图像输出（默认值0：不输出；1：输出），重启程序后恢复默认值
                                                                               #~english Whether to output rgbd image: 1 (not output, default value), 0 (output). The default value will be restored after rebooting
MV3D_RGBD_INT_DEVICE_TIMEOUT            = "DeviceTimeout"                      #~chinese 设备控制超时时间(ms)          ~english Timeout period of device control (unit: ms)
MV3D_RGBD_INT_IMAGE_NODE_NUM            = "ImageNodeNum"                       #~chinese 图像缓存节点个数              ~english The number of image buffer node
MV3D_RGBD_FLOAT_Z_UNIT                  = "ZUnit"                              #~chinese 深度图量纲(mm)                ~english The dimension of depth image (unit: mm)
MV3D_RGBD_ENUM_POINT_CLOUD_OUTPUT       = "PointCloudOutput"                   #~chinese 输出点云数据类型，详见Mv3dRgbdPointCloudType      ~english The types of output point cloud. Refer to Mv3dRgbdPointCloudType
MV3D_RGBD_INT_DOWNSAMPLE                = "DownSample"                         #~chinese 降采样参数（默认值0：不输出；1：图像宽高降低一半）  ~english Down sampling param: 0 (output original data), 1 (output half width and height)

# ch:设备文件枚举常量 | en: File Constant Definition
MV3D_RGBD_SENSOR_CALI                   = "RGBDSensorCali"                     #~chinese 相机传感器标定文件            ~english The calibration file of camera sensor
MV3D_RGBD_HANDEYE_CALI                  = "RGBDHandEyeCali"                    #~chinese 相机手眼标定文件              ~english The camera hand-eye calibration file

# Mv3dRgbdDeviceType
# ch:设备类型 | en: Device Type
DeviceType_Ethernet                     = 1                                    #~chinese 网口设备                      ~english Network type camera 
DeviceType_USB                          = 2                                    #~chinese USB设备                       ~english USB type camera
DeviceType_Ethernet_Vir                 = 4                                    #~chinese 虚拟网口设备                  ~english Virtual network type camera
DeviceType_USB_Vir                      = 8                                    #~chinese 虚拟USB设备                   ~english Virtual USB type camera

# Mv3dRgbdIpCfgMode
# ch:ip类型 | en: IP Address Mode
IpCfgMode_Static                        = 1                                    #~chinese 静态IP                        ~english Static IP mode
IpCfgMode_DHCP                          = 2                                    #~chinese 自动分配IP(DHCP)              ~english Automatically assigned IP address (DHCP)
IpCfgMode_LLA                           = 4                                    #~chinese 自动分配IP(LLA)               ~english Automatically assigned IP address (LLA) 

# Mv3dRgbdUsbProtocol
# ch:USB协议 | en: Supported USB Protocol Type
UsbProtocol_USB2                        = 1                                    #~chinese USB2协议                      ~english USB 2.0
UsbProtocol_USB3                        = 2                                    #~chinese USB3协议                      ~english USB 3.0

# Mv3dRgbdImageType
# ch:图像格式 | en:Image Format 
ImageType_Undefined                     = -1                                   #~chinese 未定义                        ~english Undefined
ImageType_Mono8                         = 17301505                             #~chinese Mono8                         ~english Mono8
ImageType_Mono16                        = 17825799                             #~chinese Mono16                        ~english Mono16
ImageType_Depth                         = 17825976                             #~chinese C16                           ~english C16
ImageType_YUV422                        = 34603058                             #~chinese YUV422                        ~english YUV422
ImageType_YUV420SP_NV12                 = 34373633                             #~chinese YUV420SP_NV12                 ~english YUV420SP_NV12
ImageType_YUV420SP_NV21                 = 34373634                             #~chinese YUV420SP_NV21                 ~english YUV420SP_NV21
ImageType_RGB8_Planar                   = 35127329                             #~chinese RGB8_Planar                   ~english RGB8_Planar
ImageType_PointCloud                    = 39846080                             #~chinese ABC32f                        ~english ABC32f
ImageType_PointCloudWithNormals         = -2134900735                          #~chinese MV3D_RGBD_POINT_XYZ_NORMALS   ~english MV3D_RGBD_POINT_XYZ_NORMALS
ImageType_TexturedPointCloud            = -2139619326                          #~chinese MV3D_RGBD_POINT_XYZ_RGB       ~english MV3D_RGBD_POINT_XYZ_RGB
ImageType_TexturedPointCloudWithNormals = -2133327869                          #~chinese MV3D_RGBD_POINT_XYZ_RGB_NORMALS        ~english MV3D_RGBD_POINT_XYZ_RGB_NORMALS
ImageType_Jpeg                          = -2145910783                          #~chinese Jpeg                          ~english Jpeg
ImageType_Rgbd                          = -2111295481                          #~chinese RGBD                          ~english RGBD

# Mv3dRgbdStreamType
# ch:数据流类型 | en: Data Stream Type
StreamType_Undefined                    = 0
StreamType_Depth                        = 1                                    #~chinese 深度图数据流                   ~english Depth image data stream
StreamType_Color                        = 2                                    #~chinese Color图数据流                 ~english Color image data stream 
StreamType_Ir_Left                      = 3                                    #~chinese 矫正后的左目图数据流           ~english Corrected left-eye image data stream
StreamType_Ir_Right                     = 4                                    #~chinese 矫正后的右目图数据流           ~english Corrected right-eye image data stream
StreamType_Imu                          = 5                                    #~chinese IMU数据流                     ~english IMU data stream
StreamType_LeftMono                     = 6                                    #~chinese 左目泛光图数据流               ~english Illuminated left-eye image data stream
StreamType_Mask                         = 7                                    #~chinese 掩膜图数据流                   ~english Mask image data stream
StreamType_Mono                         = 8                                    #~chinese 未矫正的左右目融合图数据流      ~english Uncorrected left and right eyes fusion image data stream
StreamType_Phase                        = 9                                    #~chinese 相位图数据流                  ~english Phase image data stream
StreamType_Rgbd                         = 10                                   #~chinese RGB-D图数据流                 ~english RGB-D image data stream

# Mv3dRgbdCoordinateType
# ch:坐标系类型 | en:Coordinates Type
CoordinateType_Undefined                = 0                                    #~chinese 坐标系未定义                  ~english Undefined coordinates
CoordinateType_Depth                    = 1                                    #~chinese 深度图坐标系                  ~english Depth image coordinates
CoordinateType_RGB                      = 2                                    #~chinese RGB图坐标系                   ~english RGB image coordinates

# Mv3dRgbdDevException
# ch:异常信息 | en:Exception Information
DevException_Disconnect                 = 1                                    #~chinese 设备断开连接                  ~english The device is disconnected

# Mv3dRgbdParamType
# ch:参数类型 | en:Parameter Data Type
ParamType_Bool                          = 1                                    #~chinese Bool类型参数                  ~english Boolean
ParamType_Int                           = 2                                    #~chinese Int类型参数                   ~english Int
ParamType_Float                         = 3                                    #~chinese Float类型参数                 ~english Float
ParamType_Enum                          = 4                                    #~chinese Enum类型参数                  ~english Enumeration
ParamType_String                        = 5                                    #~chinese String类型参数                ~english String

# Mv3dRgbdFileType
# ch:深度图与rgb图存图格式 | en: Format of Saving Depth Images and RGB Images
FileType_BMP                            = 1                                    #~chinese BMP格式                       ~english BMP
FileType_JPG                            = 2                                    #~chinese JPG格式                       ~english JPG
FileType_TIFF                           = 3                                    #~chinese TIFF格式                      ~english TIFF

# Mv3dRgbdPointCloudFileType
# ch:点云图存图格式 | en:Formats of Saving Point Cloud Images
PointCloudFileType_PLY                  = 1                                    #~chinese PLY_ASCII格式                 ~english PLY(ascii)
PointCloudFileType_CSV                  = 2                                    #~chinese CSV格式                       ~english CSV
PointCloudFileType_OBJ                  = 3                                    #~chinese OBJ格式                       ~english OBJ
PointCloudFileType_PLY_Binary           = 4                                    #~chinese PLY_BINARY格式                ~english PLY(binary)
PointCloudFileType_PCD_ASCII            = 5                                    #~chinese PCD_ASCII格式                 ~english PCD(ascii)
PointCloudFileType_PCD_Binary           = 6                                    #~chinese PCD_BINARY格式                ~english PCD(binary)

# Mv3dRgbdPointCloudType
# ch:输出点云图像类型 | en:Types of Output Point Cloud Data
PointCloudType_Undefined                = 0                                    #~chinese 不输出点云                    ~english Output without point cloud
PointCloudType_Common                   = 1                                    #~chinese 点云图像                      ~english Point cloud image
PointCloudType_Normals                  = 2                                    #~chinese 带法向量信息的点云图像        ~english Point cloud image with normals
PointCloudType_Texture                  = 3                                    #~chinese 纹理点云图像                  ~english Textured point cloud image
PointCloudType_Texture_Normals          = 4                                    #~chinese 带法向量的纹理点云图像        ~english Textured point cloud image with normals

# Mv3dRgbdConvertColorMapMode
# ch:伪彩图转换映射模式 | en:Convert Color Image Mapping Mode
ConvertColorMapMode_Rainbow             = 1                                    #~chinese 彩色                          ~english Rainbow
ConvertColorMapMode_Dark_Rainbow        = 2                                    #~chinese 暗彩色                        ~english Dark rainbow
ConvertColorMapMode_Dark_Green          = 3                                    #~chinese 暗绿色                        ~english Dark green
ConvertColorMapMode_Pinkish_Red         = 4                                    #~chinese 粉红色                        ~english Pinkish red
ConvertColorMapMode_Yellow              = 5                                    #~chinese 黄色                          ~english Yellow
ConvertColorMapMode_Gray_Scale          = 6                                    #~chinese 灰度                          ~english Gray scale

# Mv3dRgbdConvertColorRangeMode
# ch:伪彩图转换范围模式 | en:Convert Color Image Range Mode
ConvertColorRangeMode_Auto              = 0                                    #~chinese 自动                          ~english Auto
ConvertColorRangeMode_Abs               = 1                                    #~chinese 绝对值                        ~english Absolute value
ConvertColorRangeMode_Percentage        = 2                                    #~chinese 百分比                        ~english Percentage

# Mv3dRgbdFunctionType
# ch:功能类型 | en:Function Type
FunctionType_TriggerId                  = 1                                    #~chinese 触发id                        ~english Trigger ID

# ch:版本信息 | en:SDK Version Information
class _MV3D_RGBD_VERSION_INFO_(Structure):
    pass
_MV3D_RGBD_VERSION_INFO_._fields_ = [
    ('nMajor', c_uint),                                                        #~chinese 主版本                        ~english The main version
    ('nMinor', c_uint),                                                        #~chinese 次版本                        ~english The minor version
    ('nRevision', c_uint),                                                     #~chinese 修正版本                      ~english The revision version
]
MV3D_RGBD_VERSION_INFO = _MV3D_RGBD_VERSION_INFO_

# ch:网口设备信息 | en:Network Type Device Information
class _MV3D_RGBD_DEVICE_NET_INFO_(Structure):
    pass
Mv3dRgbdIpCfgMode = c_int # enum

_MV3D_RGBD_DEVICE_NET_INFO_._fields_ = [
    ('chMacAddress', c_ubyte * 8),                                             #~chinese Mac地址                       ~english MAC address
    ('enIPCfgMode', Mv3dRgbdIpCfgMode),                                        #~chinese 当前IP类型                    ~english Current IP type
    ('chCurrentIp', c_ubyte * 16),                                             #~chinese 设备当前IP                    ~english Device‘s IP address
    ('chCurrentSubNetMask', c_ubyte * 16),                                     #~chinese 设备当前子网掩码              ~english Device’s subnet mask
    ('chDefultGateWay', c_ubyte * 16),                                         #~chinese 设备默认网关                  ~english Device‘s default gateway
    ('chNetExport', c_ubyte * 16),                                             #~chinese 网口IP地址                    ~english Network interface IP address
    ('nReserved', c_byte * 16),                                                #~chinese 保留字节                      ~english Reserved
]
MV3D_RGBD_DEVICE_NET_INFO = _MV3D_RGBD_DEVICE_NET_INFO_

# ch:USB设备信息 | en:USB Type Device Information
class _MV3D_RGBD_DEVICE_USB_INFO_(Structure):
    pass
Mv3dRgbdUsbProtocol = c_int # enum

_MV3D_RGBD_DEVICE_USB_INFO_._fields_ = [
    ('nVendorId', c_uint),                                                     #~chinese 供应商ID号                    ~english Manufacturer/vendor ID
    ('nProductId', c_uint),                                                    #~chinese 产品ID号                      ~english Product ID
    ('enUsbProtocol', Mv3dRgbdUsbProtocol),                                    #~chinese 支持的USB协议                 ~english Supported USB protocol types
    ('chDeviceGUID', c_ubyte * 64),                                            #~chinese 设备GUID号                    ~english Device GUID
    ('nReserved', c_byte * 16),                                                #~chinese 保留字节                      ~english Reserved
]
MV3D_RGBD_DEVICE_USB_INFO = _MV3D_RGBD_DEVICE_USB_INFO_

# ch:枚举相关设备信息 | en:Device Information
class _MV3D_RGBD_DEVICE_INFO_(Structure):
    pass
class MV3D_RGBD_DEVICE_SPECIAL_INFO(Union):
    pass
Mv3dRgbdDeviceType = c_int # enum

# ch:不同设备特有信息 | en:Particular information of different types devices
MV3D_RGBD_DEVICE_SPECIAL_INFO._fields_ = [
    ('stNetInfo', MV3D_RGBD_DEVICE_NET_INFO),                                  #~chinese 网口设备特有                  ~english Network type device
    ('stUsbInfo', MV3D_RGBD_DEVICE_USB_INFO),                                  #~chinese USB设备特有                   ~english USB type device information
]

_MV3D_RGBD_DEVICE_INFO_._fields_ = [
    ('chManufacturerName', c_ubyte * 32),                                      #~chinese 设备厂商                      ~english Manufacturer
    ('chModelName', c_ubyte * 32),                                             #~chinese 设备型号                      ~english Device model
    ('chDeviceVersion', c_ubyte * 32),                                         #~chinese 设备版本                      ~english Device version
    ('chManufacturerSpecificInfo', c_ubyte * 44),                              #~chinese 设备厂商特殊信息              ~english The specific information about manufacturer
    ('nDevTypeInfo', c_uint),                                                  #~chinese 设备类型信息                  ~english Device type info
    ('chSerialNumber', c_ubyte * 16),                                          #~chinese 设备序列号                    ~english Device serial number 
    ('chUserDefinedName', c_ubyte * 16),                                       #~chinese 设备用户自定义名称            ~english User-defined name of device
    ('enDeviceType', Mv3dRgbdDeviceType),                                      #~chinese 设备类型：网口、USB、虚拟网口设备、虚拟USB设备 
                                                                               #~english Device type(network / USB / virtual network / virtual USB)
    ('SpecialInfo', MV3D_RGBD_DEVICE_SPECIAL_INFO),                            #~chinese 不同设备特有信息              ~english Particular information of different types devices
]
MV3D_RGBD_DEVICE_INFO = _MV3D_RGBD_DEVICE_INFO_

# ch:IP配置 | en:IP Configuration Parameters
class _MV3D_RGBD_IP_CONFIG_(Structure):
    pass
Mv3dRgbdIpCfgMode = c_int # enum

_MV3D_RGBD_IP_CONFIG_._fields_ = [
    ('enIPCfgMode', Mv3dRgbdIpCfgMode),                                        #~chinese IP配置模式                    ~english IP configuration mode
    ('chDestIp', c_ubyte * 16),                                                #~chinese 设置的目标IP,仅静态IP模式下有效         ~english The IP address which is to be attributed to the target device. It is valid in the static IP mode only
    ('chDestNetMask', c_ubyte * 16),                                           #~chinese 设置的目标子网掩码,仅静态IP模式下有效   ~english The subnet mask of target device. It is valid in the static IP mode only
    ('chDestGateWay', c_ubyte * 16),                                           #~chinese 设置的目标网关,仅静态IP模式下有效       ~english The gateway of target device. It is valid in the static IP mode only

    ('nReserved', c_byte * 16),                                                #~chinese 保留字节                      ~english Reserved
]
MV3D_RGBD_IP_CONFIG = _MV3D_RGBD_IP_CONFIG_

# ch:相机图像 | en:Camera Image Parameters
class _MV3D_RGBD_IMAGE_DATA_(Structure):
    pass
Mv3dRgbdImageType = c_int # enum
Mv3dRgbdStreamType = c_int # enum
Mv3dRgbdCoordinateType = c_int # enum

_MV3D_RGBD_IMAGE_DATA_._fields_ = [
    ('enImageType', Mv3dRgbdImageType),                                        #~chinese 图像格式                      ~english Image format
    ('nWidth', c_uint),                                                        #~chinese 图像宽                        ~english Image width
    ('nHeight', c_uint),                                                       #~chinese 图像高                        ~english Image height
    ('pData', POINTER(c_ubyte)),                                               #~chinese 相机输出的图像数据            ~english Image data, which is outputted by the camera
    ('nDataLen', c_uint),                                                      #~chinese 图像数据长度(字节)            ~english Image data length (bytes) 
    ('nFrameNum', c_uint),                                                     #~chinese 帧号，代表第几帧图像          ~english Frame number, which indicates the frame sequence
    ('nTimeStamp', int64_t),                                                   #~chinese 设备上报的时间戳 （设备上电从0开始，规则详见设备手册） 
                                                                               #~english Timestamp uploaded by the device. It starts from 0 when the device is powered on. Refer to the device user manual for detailed rules
    ('bIsRectified', c_uint),                                                  #~chinese 是否校正                      ~english Correction flag 
    ('enStreamType', Mv3dRgbdStreamType),                                      #~chinese 流类型，用于区分图像(图像格式相同时)    ~english Data stream type, used to distinguish data in the same image format
    ('enCoordinateType', Mv3dRgbdCoordinateType),                              #~chinese 坐标系类型                    ~english Coordinates type

    ('nReserved', c_byte * 4),                                                 #~chinese 保留字节                      ~english Reserved
]
MV3D_RGBD_IMAGE_DATA = _MV3D_RGBD_IMAGE_DATA_

class _MV3D_RGBD_FRAME_DATA_(Structure):
    pass
# ch:图像帧数据 | en:Frame Data
_MV3D_RGBD_FRAME_DATA_._fields_ = [
    ('nImageCount', c_uint),                                                   #~chinese 图像个数，表示stImage数组的有效个数     ~english The number of images. It indicates the number of valid stImage arrays
    ('stImageData', MV3D_RGBD_IMAGE_DATA * MV3D_RGBD_MAX_IMAGE_COUNT),         #~chinese 图像数组，每一个代表一种类型的图像      ~english Image array, each one represents one type of images
    ('nValidInfo', c_uint),                                                    #~chinese 帧有效信息：0（帧有效），1 << 0（丢包），1 << 1（触发标识符无效）
                                                                               #~english Frame valid info: 0 (Frame is valid), 1 << 0 (lost package), 1 << 1 (trigger is not valid)
    ('nTriggerId', c_uint),                                                    #~chinese 触发标记                      ~english Trigger ID
    ('nReserved', c_byte * 8),                                                 #~chinese 保留字节                      ~english Reserved
]
MV3D_RGBD_FRAME_DATA = _MV3D_RGBD_FRAME_DATA_

# ch:float格式2维点数据 | en:Float type two-dimension point data
class _MV3D_RGBD_POINT_2D_F32_(Structure):
    pass
_MV3D_RGBD_POINT_2D_F32_._fileds_ = [
    ('fX', c_float),                                                           #~chinese X轴坐标                      ~english X-dimension data 
    ('fY', c_float),                                                           #~chinese Y轴坐标                      ~english Y-dimension data 
]
MV3D_RGBD_POINT_2D_F32 = _MV3D_RGBD_POINT_2D_F32_

# ch:float格式3维点数据 | en:Float type two-dimension point data
class _MV3D_RGBD_POINT_3D_F32_(Structure):
    pass
_MV3D_RGBD_POINT_3D_F32_._fileds_ = [
    ('fX', c_float),                                                           #~chinese X轴坐标                      ~english X-dimension data 
    ('fY', c_float),                                                           #~chinese Y轴坐标                      ~english Y-dimension data 
    ('fZ', c_float),                                                           #~chinese Z轴坐标                      ~english Z-dimension data 
]
MV3D_RGBD_POINT_3D_F32 = _MV3D_RGBD_POINT_3D_F32_

# ch:法向量点云数据 | en:Point cloud with normal vector data
class _MV3D_RGBD_POINT_XYZ_NORMALS_(Structure):
    pass
_MV3D_RGBD_POINT_XYZ_NORMALS_._fileds_ = [
    ('stPoint3f', MV3D_RGBD_POINT_3D_F32),                                     #~chinese 点云数据                     ~english Point cloud data
    ('stNormals', MV3D_RGBD_POINT_3D_F32),                                     #~chinese 法向量数据                   ~english Normal vector data
]
MV3D_RGBD_POINT_XYZ_NORMALS = _MV3D_RGBD_POINT_XYZ_NORMALS_

# ch:rgba数据 | en:rgba data
class _MV3D_RGBD_POINT_RGBA_(Structure):
    pass
_MV3D_RGBD_POINT_RGBA_._fileds_ = [
    ('nR', c_uint8),                                                           #~chinese R通道数据                    ~english R channel data
    ('nG', c_uint8),                                                           #~chinese G通道数据                    ~english G channel data
    ('nB', c_uint8),                                                           #~chinese B通道数据                    ~english B channel data 
    ('nA', c_uint8),                                                           #~chinese A通道数据                    ~english A channel data
]
MV3D_RGBD_POINT_RGBA = _MV3D_RGBD_POINT_RGBA_

# ch:rgb信息 | en:rgb info
class _MV3D_RGBD_RGB_INFO_(Union):
    pass
_MV3D_RGBD_RGB_INFO_._fileds_ = [
    ('stRgba', MV3D_RGBD_POINT_RGBA),                                          #~chinese rgba信息                    ~english rgba info
    ('fRgb', c_float),                                                         #~chinese float格式rgb信息             ~english float type rgb info
]
MV3D_RGBD_RGB_INFO = _MV3D_RGBD_RGB_INFO_

# ch:纹理点云数据 | en:Textured point cloud data
class _MV3D_RGBD_POINT_XYZ_RGB_(Structure):
    pass
_MV3D_RGBD_POINT_XYZ_RGB_._fields_ = [
    ('stPoint3f', MV3D_RGBD_POINT_3D_F32),                                     #~chinese 点云数据                     ~english Point cloud data
    ('stRgbInfo', MV3D_RGBD_RGB_INFO),                                         #~chinese rgb信息                      ~english rgb info
]
MV3D_RGBD_POINT_XYZ_RGB = _MV3D_RGBD_POINT_XYZ_RGB_

# ch:带法向量的纹理点云数据 | en:Textured point cloud with normal vector data
class _MV3D_RGBD_POINT_XYZ_RGB_NORMALS_(Structure):
    pass
_MV3D_RGBD_POINT_XYZ_RGB_NORMALS_._fields_ = [
    ('stRgbPoint', MV3D_RGBD_POINT_XYZ_RGB),                                   #~chinese 彩色点云数据                 ~english Color point cloud data
    ('stNormals', MV3D_RGBD_POINT_3D_F32),                                     #~chinese 法向量数据                   ~english Normal vector data
]
MV3D_RGBD_POINT_XYZ_RGB_NORMALS = _MV3D_RGBD_POINT_XYZ_RGB_NORMALS_

# ch: UV图数据  | en:english UV map data
class _MV3D_RGBD_UV_DATA_(Structure):
    pass
_MV3D_RGBD_UV_DATA_._fields_ = [ 
    ('nDataLen',c_uint),                                                       #~chinese 数据长度                      ~english Data length
    ('pData',POINTER(MV3D_RGBD_POINT_2D_F32)),                                 #~chinese UV图数据                      ~english UV Map data

    ('nReserved', c_byte * 8),                                                 #~chinese 保留字节                      ~english Reserved
]
MV3D_RGBD_UV_DATA = _MV3D_RGBD_UV_DATA_

#~ch: 深度图伪彩转换参数 | en: Depth to Color Image Conversion Parameters
class _MV3D_RGBD_CONVERT_COLOR_PAPRAM_(Structure):
    pass
Mv3dRgbdConvertColorMapMode = c_int # enum
Mv3dRgbdConvertColorRangeMode = c_int # enum
_MV3D_RGBD_CONVERT_COLOR_PAPRAM_._fields_ = [ 
    ('enConvertColorMapMode',Mv3dRgbdConvertColorMapMode),                     #~chinese 颜色映射模式                  ~english Color mapping mode
    ('enConvertColorRangeMode',Mv3dRgbdConvertColorRangeMode),                 #~chinese 深度图转伪彩范围模式          ~english Depth Image to Color Image Range Mode
    ('bExistInvalidValue',c_bool),                                             #~chinese 无效值数量                    ~english The number of invalid values
    ('fInvalidValue',c_float * 8),                                             #~chinese 无效值                        ~english Invalid values
    ('fRangeMinValue',c_float),                                                #~chinese 范围模式为用户指定绝对值时，表示最小深度值；范围模式为用户指定百分比时，表示最小百分比[0.0,1.0]
                                                                               #~english If the absolute value is chosen as the range mode, it represents the minimum depth value. Else if the percentage is chosen as the range mode, it represents the minimum percentage
    ('fRangeMaxValue',c_float),                                                #~chinese 范围模式为用户指定绝对值时，表示最大深度值；范围模式为用户指定百分比时，表示最大百分比[0.0,1.0]
                                                                               #~english If the absolute value is chosen as the range mode, it represents the maximum depth value. Else if the percentage is chosen as the range mode, it represents the maximum percentage
    ('nReserved', c_byte * 8),                                                 #~chinese 保留字节                      ~english Reserved
]
MV3D_RGBD_CONVERT_COLOR_PAPRAM = _MV3D_RGBD_CONVERT_COLOR_PAPRAM_

# ch:固件输出的图像附加信息 | en:Image Additional Information Output by Firmware
class _MV3D_RGBD_STREAM_CFG_(Structure):
    pass
Mv3dRgbdImageType = c_int # enum

_MV3D_RGBD_STREAM_CFG_._fields_ = [
    ('enImageType', Mv3dRgbdImageType),                                        #~chinese 图像格式                      ~english Image format
    ('nWidth', c_uint),                                                        #~chinese 图像宽                        ~english Image width
    ('nHeight', c_uint),                                                       #~chinese 图像高                        ~english Image height

    ('nReserved', c_byte * 32),                                                #~chinese 保留字节                      ~english Reserved
]
MV3D_RGBD_STREAM_CFG = _MV3D_RGBD_STREAM_CFG_

# ch:固件输出的图像帧附加信息 | en:Frame Additional Information Output by Firmware
class _MV3D_RGBD_STREAM_CFG_LIST_(Structure):
    pass
_MV3D_RGBD_STREAM_CFG_LIST_._fields_ = [
    ('nStreamCfgCount', c_uint),                                               #~chinese 图像信息数量                  ~english The number of image information
    ('stStreamCfg', MV3D_RGBD_STREAM_CFG * MV3D_RGBD_MAX_IMAGE_COUNT),         #~chinese 图像附加信息                  ~english Image additional information

    ('nReserved', c_byte * 16),                                                #~chinese 保留字节                      ~english Reserved
]
MV3D_RGBD_STREAM_CFG_LIST = _MV3D_RGBD_STREAM_CFG_LIST_

class _MV3D_RGBD_DEVICE_INFO_LIST_(Structure):
    pass
_MV3D_RGBD_DEVICE_INFO_LIST_._fields_ = [
    ('DeviceInfo', MV3D_RGBD_DEVICE_INFO * 20),                                #~chinese 设备信息结构体数组，目前最大20
                                                                               #~english Device information struct, max number is 20
]
MV3D_RGBD_DEVICE_INFO_LIST = _MV3D_RGBD_DEVICE_INFO_LIST_

# ch:相机内参，3x3 matrix | en: Camera Internal Parameters
# | fx|  0| cx|
# |  0| fy| cy|
# |  0|  0|  1|
class _MV3D_RGBD_CAMERA_INTRINSIC_(Structure):
    pass
_MV3D_RGBD_CAMERA_INTRINSIC_._fields_ = [
    ('fData', c_float * 9),                                                    #~chinese 内参参数：fx,0,cx,0,fy,cy,0,0,1         
                                                                               #~english Internal parameters: fx,0,cx,0,fy,cy,0,0,1
]
MV3D_RGBD_CAMERA_INTRINSIC = _MV3D_RGBD_CAMERA_INTRINSIC_

class _MV3D_RGBD_CAMERA_DISTORTION_(Structure):
    pass
_MV3D_RGBD_CAMERA_DISTORTION_._fields_ = [
    ('fData', c_float * 12),                                                   #~chinese 畸变系数：k1,k2,p1,p2,k3,k4,k5,k6,s1,s2,s3,s4    
                                                                               #~english Distortion coefficient: k1,k2,p1,p2,k3,k4,k5,k6,s1,s2,s3,s4
]
MV3D_RGBD_CAMERA_DISTORTION = _MV3D_RGBD_CAMERA_DISTORTION_

class _MV3D_RGBD_CALIB_INFO_(Structure):
    pass
_MV3D_RGBD_CALIB_INFO_._fields_ = [
    ('stIntrinsic', MV3D_RGBD_CAMERA_INTRINSIC),                               #~chinese 相机内参                      ~english Camera internal parameters
    ('stDistortion', MV3D_RGBD_CAMERA_DISTORTION),                             #~chinese 畸变系数                      ~english Camera distortion coefficient
    ('nWidth', c_uint),                                                        #~chinese 图像宽                        ~english Image width
    ('nHeight', c_uint),                                                       #~chinese 图像高                        ~english Image height

    ('nReserved', c_byte * 8),                                                 #~chinese 保留字节                      ~english Reserved
]
MV3D_RGBD_CALIB_INFO = _MV3D_RGBD_CALIB_INFO_

# ch:相机深度图转Rgb的外参，4x4 matrix | en:Camera Extrinsic Parameters of Depth Image to Rgb Image
# | r00| r01| r02| t0|  
# | r10| r11| r12| t1|
# | r20| r21| r22| t2|
# |   0|   0|   0|  1|
class _MV3D_RGBD_CAMERA_EXTRINSIC_(Structure):
    pass
_MV3D_RGBD_CAMERA_EXTRINSIC_._fields_ = [
    ('fData', c_float * 16),                                                   #~chinese 深度图转Rgb外参参数：r00,r01,r02,t0,r10,r11,r12,t1,r20,r21,r22,t2,0,0,0,1 
                                                                               #~english Extrinsic parameters of depth image to rgb image: r00,r01,r02,t0,r10,r11,r12,t1,r20,r21,r22,t2,0,0,0,1
]
MV3D_RGBD_CAMERA_EXTRINSIC = _MV3D_RGBD_CAMERA_EXTRINSIC_

# ch:相机参数信息 | en:Camera Parameters Information
class _MV3D_RGBD_CAMERA_PARAM_(Structure):
    pass
_MV3D_RGBD_CAMERA_PARAM_._fields_ = [
    ('stDepthCalibInfo', MV3D_RGBD_CALIB_INFO),                                #~chinese 深度图内参和畸变矩阵信息      ~english Depth image intrinsic information and distortion coefficient
    ('stRgbCalibInfo', MV3D_RGBD_CALIB_INFO),                                  #~chinese rgb内参和畸变矩阵信息         ~english Rgb image intrinsic information and distortion coefficient
    ('stDepth2RgbExtrinsic', MV3D_RGBD_CAMERA_EXTRINSIC),                      #~chinese 相机深度图转RGB的外参         ~english Camera extrinsic parameters of depth image to rgb image
    ('nReserved', c_byte * 32),                                                #~chinese 保留字节                      ~english Reserved
]
MV3D_RGBD_CAMERA_PARAM = _MV3D_RGBD_CAMERA_PARAM_

# ch:Int类型值 | en:Int Type Value
class _MV3D_RGBD_INTPARAM_(Structure):
    pass
_MV3D_RGBD_INTPARAM_._fields_ = [
    ('nCurValue', int64_t),                                                    #~chinese 当前值                        ~english Current value
    ('nMax', int64_t),                                                         #~chinese 最大值                        ~english The maximum value
    ('nMin', int64_t),                                                         #~chinese 最小值                        ~english The minimum value
    ('nInc', int64_t),                                                         #~chinese 增量值                        ~english The increment value
]
MV3D_RGBD_INTPARAM = _MV3D_RGBD_INTPARAM_

# ch:Enum类型值 | en:Enumeration Type Value
class _MV3D_RGBD_ENUMPARAM_(Structure):
    pass
_MV3D_RGBD_ENUMPARAM_._fields_ = [
    ('nCurValue', c_uint),                                                     #~chinese 当前值                        ~english Current value
    ('nSupportedNum', c_uint),                                                 #~chinese 有效数据个数                  ~english The number of valid data
    ('nSupportValue', c_uint * MV3D_RGBD_MAX_ENUM_COUNT),                      #~chinese 支持的枚举类型                ~english The type of supported enumerations
]
MV3D_RGBD_ENUMPARAM = _MV3D_RGBD_ENUMPARAM_

# ch:Float类型值 | en:Float Type Value
class _MV3D_RGBD_FLOATPARAM_(Structure):
    pass
_MV3D_RGBD_FLOATPARAM_._fields_ = [
    ('fCurValue', c_float),                                                    #~chinese 当前值                        ~english Current value
    ('fMax', c_float),                                                         #~chinese 最大值                        ~english The maximum value
    ('fMin', c_float),                                                         #~chinese 最小值                        ~english The minimum value
]
MV3D_RGBD_FLOATPARAM = _MV3D_RGBD_FLOATPARAM_

# ch:String类型值 | en:String Type Value
class _MV3D_RGBD_STRINGPARAM_(Structure):
    pass
_MV3D_RGBD_STRINGPARAM_._fields_ = [
    ('chCurValue', c_char * MV3D_RGBD_MAX_STRING_LENGTH),                      #~chinese 当前值                        ~english Current value
    ('nMaxLength', uint32_t),                                                  #~chinese 属性节点能设置字符的最大长度  ~english The maximum length of string
]
MV3D_RGBD_STRINGPARAM = _MV3D_RGBD_STRINGPARAM_

# ch:设备参数值 | en:Device Parameters
class _MV3D_RGBD_PARAM_(Structure):
    pass
    
Mv3dRgbdParamType = c_int # enum

class MV3D_RGBD_PARAM_ParamInfo(Union):
    pass
MV3D_RGBD_PARAM_ParamInfo._fields_=[
    ('bBoolParam', c_bool),                                                    #~chinese Bool类型参数                  ~english Boolean type parameter
    ('stIntParam', MV3D_RGBD_INTPARAM),                                        #~chinese Int类型参数                   ~english Int type parameter
    ('stFloatParam', MV3D_RGBD_FLOATPARAM),                                    #~chinese Float类型参数                 ~english Float type parameter
    ('stEnumParam', MV3D_RGBD_ENUMPARAM),                                      #~chinese Enum类型参数                  ~english Enum type parameter
    ('stStringParam', MV3D_RGBD_STRINGPARAM),                                  #~chinese String类型参数                ~english String type parameter
]

_MV3D_RGBD_PARAM_._fields_=[
    ('enParamType', Mv3dRgbdParamType),
    ('ParamInfo', MV3D_RGBD_PARAM_ParamInfo),
    ('nReserved', c_byte*16),
]
MV3D_RGBD_PARAM = _MV3D_RGBD_PARAM_

# ch:异常信息 | en:Exception Information
class _MV3D_RGBD_EXCEPTION_INFO_(Structure):
    pass
Mv3dRgbdDevException = c_int # enum
_MV3D_RGBD_EXCEPTION_INFO_._fields_=[
    ('enExceptionId',Mv3dRgbdDevException),                                    #~chinese 异常ID                        ~english Exception ID
    ('chExceptionDes',c_char*MV3D_RGBD_MAX_STRING_LENGTH),                     #~chinese 异常描述                      ~english Exception description
    ('nReserved',c_byte*4),                                                    #~chinese 保留字节                      ~english Reserved
]
MV3D_RGBD_EXCEPTION_INFO=_MV3D_RGBD_EXCEPTION_INFO_

# ch:文件存取 | en:File Access
class _MV3D_RGBD_FILE_ACCESS_(Structure):
    pass
_MV3D_RGBD_FILE_ACCESS_._fields_ = [
    ('pUserFileName', c_char_p),                                               #~chinese 用户文件名                    ~english User file name
    ('pDevFileName', c_char_p),                                                #~chinese 设备文件名                    ~english Device file name
    
    ('nReserved', c_byte*32),                                                  #~chinese 保留字节                      ~english Reserved
]
MV3D_RGBD_FILE_ACCESS = _MV3D_RGBD_FILE_ACCESS_ 

# ch:文件存取进度 | en:File Access Progress
class _MV3D_RGBD_FILE_ACCESS_PROGRESS_(Structure):
    pass
_MV3D_RGBD_FILE_ACCESS_PROGRESS_._fields_ = [
    ('nCompleted', c_int64),                                                   #~chinese 已完成的长度                  ~english Completed length
    ('nTotal', c_int64),                                                       #~chinese 总长度                        ~english Total length

    ('nReserved', c_byte*32),                                                  #~chinese 保留字节                      ~english Reserved
]
MV3D_RGBD_FILE_ACCESS_PROGRESS = _MV3D_RGBD_FILE_ACCESS_PROGRESS_

__all__ = ['_MV3D_RGBD_VERSION_INFO_','MV3D_RGBD_VERSION_INFO','_MV3D_RGBD_DEVICE_NET_INFO_','MV3D_RGBD_DEVICE_NET_INFO','_MV3D_RGBD_DEVICE_USB_INFO_','MV3D_RGBD_DEVICE_USB_INFO', 
           '_MV3D_RGBD_DEVICE_INFO_','MV3D_RGBD_DEVICE_INFO','_MV3D_RGBD_IP_CONFIG_','MV3D_RGBD_IP_CONFIG','_MV3D_RGBD_IMAGE_DATA_','MV3D_RGBD_IMAGE_DATA',
           '_MV3D_RGBD_FRAME_DATA_','MV3D_RGBD_FRAME_DATA','_MV3D_RGBD_STREAM_CFG_','MV3D_RGBD_STREAM_CFG','_MV3D_RGBD_STREAM_CFG_LIST_','MV3D_RGBD_STREAM_CFG_LIST',
           '_MV3D_RGBD_DEVICE_INFO_LIST_', 'MV3D_RGBD_DEVICE_INFO_LIST','_MV3D_RGBD_CAMERA_INTRINSIC_','MV3D_RGBD_CAMERA_INTRINSIC','_MV3D_RGBD_CAMERA_DISTORTION_','MV3D_RGBD_CAMERA_DISTORTION',
           '_MV3D_RGBD_CALIB_INFO_','MV3D_RGBD_CALIB_INFO','_MV3D_RGBD_CAMERA_EXTRINSIC_','MV3D_RGBD_CAMERA_EXTRINSIC','_MV3D_RGBD_CAMERA_PARAM_','MV3D_RGBD_CAMERA_PARAM',
           '_MV3D_RGBD_INTPARAM_', 'MV3D_RGBD_INTPARAM','_MV3D_RGBD_ENUMPARAM_','MV3D_RGBD_ENUMPARAM','_MV3D_RGBD_FLOATPARAM_','MV3D_RGBD_FLOATPARAM',
           '_MV3D_RGBD_STRINGPARAM_','MV3D_RGBD_STRINGPARAM','_MV3D_RGBD_PARAM_','MV3D_RGBD_PARAM','_MV3D_RGBD_EXCEPTION_INFO_','MV3D_RGBD_EXCEPTION_INFO','_MV3D_RGBD_POINT_RGBA_','MV3D_RGBD_POINT_RGBA',
           '_MV3D_RGBD_RGB_INFO_','MV3D_RGBD_RGB_INFO','_MV3D_RGBD_FILE_ACCESS_','MV3D_RGBD_FILE_ACCESS', '_MV3D_RGBD_FILE_ACCESS_PROGRESS_', 'MV3D_RGBD_FILE_ACCESS_PROGRESS',
           '_MV3D_RGBD_POINT_2D_F32_', 'MV3D_RGBD_POINT_2D_F32', '_MV3D_RGBD_POINT_3D_F32_','MV3D_RGBD_POINT_3D_F32','_MV3D_RGBD_POINT_XYZ_NORMALS_','MV3D_RGBD_POINT_XYZ_NORMALS',
           '_MV3D_RGBD_POINT_XYZ_RGB_','MV3D_RGBD_POINT_XYZ_RGB', '_MV3D_RGBD_POINT_XYZ_RGB_NORMALS_','MV3D_RGBD_POINT_XYZ_RGB_NORMALS','_MV3D_RGBD_UV_DATA_', 'MV3D_RGBD_UV_DATA',
           '_MV3D_RGBD_CONVERT_COLOR_PAPRAM_', 'MV3D_RGBD_CONVERT_COLOR_PAPRAM']