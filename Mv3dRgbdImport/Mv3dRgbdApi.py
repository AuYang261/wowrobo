# -- coding: utf-8 --

import sys
import copy
import ctypes
import os

from ctypes import *

# ch: python3.8及相关版本可以指定如下路径的库 | en: python3.8 and related versions can specify libraries for the following paths
os.add_dll_directory("C:\Program Files (x86)\Common Files\MV3D\Runtime\Win64_x64")
os.add_dll_directory(
    "C:\Program Files (x86)\Common Files\Mv3dRgbdSDK\Runtime\Win64_x64"
)

Mv3dRgbdDll = ctypes.WinDLL("Mv3dRgbd.dll")


class Mv3dRgbd:
    def __init__(self):
        self._handle = (
            c_void_p()
        )  # ~chinese 记录当前连接设备的句柄        ~english Records the handle of the currently connected device
        self.handle = pointer(
            self._handle
        )  # ~chinese 创建句柄指针                  ~english Create pointer of handle

    # *******************************************************************************************************
    # brief  获取SDK版本号
    # param  pstVersion                       [OUT]           版本信息
    # return 成功，返回MV3D_RGBD_OK；错误，返回错误码

    # brief  Get SDK Version
    # param  pstVersion                       [OUT]           version info
    # return Success, return MV3D_RGBD_OK. Failure, return error code
    @staticmethod
    def MV3D_RGBD_GetSDKVersion(pstVersion):
        Mv3dRgbdDll.MV3D_RGBD_GetSDKVersion.argtype = c_void_p
        Mv3dRgbdDll.MV3D_RGBD_GetSDKVersion.restype = c_uint
        # C：MV3D_RGBD_API MV3D_RGBD_STATUS MV3D_RGBD_GetSDKVersion(MV3D_RGBD_VERSION_INFO* pstVersion);
        return Mv3dRgbdDll.MV3D_RGBD_GetSDKVersion(pstVersion)

    # *******************************************************************************************************

    # *******************************************************************************************************
    # brief  SDK运行环境初始化
    # param
    # return 成功，返回MV3D_RGBD_OK；错误，返回错误码

    # brief  SDK run environment initialization
    # param
    # return Success, return MV3D_RGBD_OK. Failure, return error code
    @staticmethod
    def MV3D_RGBD_Initialize():
        Mv3dRgbdDll.MV3D_RGBD_Initialize.restype = c_uint
        # C：MV3D_RGBD_API MV3D_RGBD_STATUS MV3D_RGBD_Initialize();
        return Mv3dRgbdDll.MV3D_RGBD_Initialize()

    # *******************************************************************************************************

    # *******************************************************************************************************
    # brief  SDK运行环境释放
    # param
    # return 成功，返回MV3D_RGBD_OK；错误，返回错误码

    # brief  SDK run environment release
    # param
    # return Success, return MV3D_RGBD_OK. Failure, return error code
    @staticmethod
    def MV3D_RGBD_Release():
        Mv3dRgbdDll.MV3D_RGBD_Release.restype = c_uint
        # C：MV3D_RGBD_API MV3D_RGBD_STATUS MV3D_RGBD_Release();
        return Mv3dRgbdDll.MV3D_RGBD_Release()

    # *******************************************************************************************************

    # *******************************************************************************************************
    # brief  获取当前环境中设备数量
    # param  nDeviceType                      [IN]            设备类型,见Mv3dRgbdDeviceType,可全部获取(DeviceType_Ethernet | DeviceType_USB | DeviceType_Ethernet_Vir | DeviceType_USB_Vir)
    # param  pDeviceNumber                    [OUT]           设备数量
    # return 成功，返回MV3D_RGBD_OK；错误，返回错误码

    # brief  gets the number of devices in the current environment
    # param  nDeviceType                      [IN]            device type，refer to Mv3dRgbdDeviceType，get all(DeviceType_Ethernet | DeviceType_USB | DeviceType_Ethernet_Vir | DeviceType_USB_Vir)
    # param  pDeviceNumber                    [OUT]           device number
    # return Success, return MV3D_RGBD_OK. Failure, return error code
    @staticmethod
    def MV3D_RGBD_GetDeviceNumber(nDeviceType, pDeviceNumber):
        Mv3dRgbdDll.MV3D_RGBD_GetDeviceNumber.argtype = (c_uint, c_char_p)
        Mv3dRgbdDll.MV3D_RGBD_GetDeviceNumber.restype = c_uint
        # C原型：MV3D_RGBD_API MV3D_RGBD_STATUS MV3D_RGBD_GetDeviceNumber(uint32_t nDeviceType, uint32_t* pDeviceNumber);
        return Mv3dRgbdDll.MV3D_RGBD_GetDeviceNumber(nDeviceType, pDeviceNumber)

    # *******************************************************************************************************

    # *******************************************************************************************************
    # brief  获取设备列表
    # param  nDeviceType                      [IN]            设备类型,见Mv3dRgbdDeviceType,可全部获取(DeviceType_Ethernet | DeviceType_USB | DeviceType_Ethernet_Vir | DeviceType_USB_Vir)
    # param  pstDeviceInfos                   [IN OUT]        设备列表
    # param  nMaxDeviceCount                  [IN]            设备列表缓存最大个数
    # param  pDeviceCount                     [OUT]           填充列表中设备个数
    # return 成功，返回MV3D_RGBD_OK；错误，返回错误码

    # brief  gets 3D cameras list
    # param  nDeviceType                      [IN]            device type，refer to Mv3dRgbdDeviceType，get all(DeviceType_Ethernet | DeviceType_USB | DeviceType_Ethernet_Vir | DeviceType_USB_Vir)
    # param  pstDeviceInfos                   [IN OUT]        devices list
    # param  nMaxDeviceCount                  [IN]            max Number of device list caches
    # param  pDeviceCount                     [OUT]           number of devices in the fill list
    # return Success, return MV3D_RGBD_OK. Failure, return error code
    @staticmethod
    def MV3D_RGBD_GetDeviceList(
        nDeviceType, pstDeviceInfos, nMaxDeviceCount, pDeviceNumber
    ):
        Mv3dRgbdDll.MV3D_RGBD_GetDeviceList.argtype = (
            c_uint,
            c_char_p,
            c_uint,
            c_char_p,
        )
        Mv3dRgbdDll.MV3D_RGBD_GetDeviceList.restype = c_uint
        # C：MV3D_RGBD_API MV3D_RGBD_STATUS MV3D_RGBD_GetDeviceList(uint32_t nDeviceType, MV3D_RGBD_DEVICE_INFO* pstDeviceInfos, uint32_t nMaxDeviceCount, uint32_t* pDeviceCount);
        return Mv3dRgbdDll.MV3D_RGBD_GetDeviceList(
            nDeviceType, pstDeviceInfos, nMaxDeviceCount, pDeviceNumber
        )

    # *******************************************************************************************************

    # *******************************************************************************************************
    # brief  打开设备
    # param  self.handle                      [IN OUT]        相机句柄
    # param  pstDeviceInfo                    [IN]            枚举的设备信息，默认为空，打开第一个相机
    # return 成功，返回MV3D_RGBD_OK；错误，返回错误码

    # brief  open device
    # param  self.handle                      [IN OUT]        camera handle
    # param  pstDeviceInfo                    [IN]            enum camera info. the default is null, open first camera
    # return Success, return MV3D_RGBD_OK. Failure, return error code
    def MV3D_RGBD_OpenDevice(self, pstDeviceInfo):
        Mv3dRgbdDll.MV3D_RGBD_OpenDevice.argtype = (c_char_p, c_char_p)
        Mv3dRgbdDll.MV3D_RGBD_OpenDevice.restype = c_uint
        # C：MV3D_RGBD_API MV3D_RGBD_STATUS MV3D_RGBD_OpenDevice(HANDLE *handle, MV3D_RGBD_DEVICE_INFO* pstDeviceInfo = NULL);
        return Mv3dRgbdDll.MV3D_RGBD_OpenDevice(byref(self.handle), pstDeviceInfo)

    # *******************************************************************************************************

    # *******************************************************************************************************
    # brief  通过设备设备自定义打开设备
    # param  self.handle                      [IN OUT]        相机句柄
    # param  chDeviceName                     [IN]            设备用户自定义名称
    # return 成功，返回MV3D_RGBD_OK；错误，返回错误码

    # brief  open device by device user defined name
    # param  self.handle                      [IN OUT]        camera handle
    # param  chDeviceName                     [IN]            device user defined name
    # return Success, return MV3D_RGBD_OK. Failure, return error code
    def MV3D_RGBD_OpenDeviceByName(self, chDeviceName):
        Mv3dRgbdDll.MV3D_RGBD_OpenDeviceByName.argtype = (c_void_p, c_void_p)
        Mv3dRgbdDll.MV3D_RGBD_OpenDeviceByName.restype = c_uint
        # C：MV3D_RGBD_API MV3D_RGBD_STATUS MV3D_RGBD_OpenDeviceByName(HANDLE *handle, const char* chDeviceName);
        return Mv3dRgbdDll.MV3D_RGBD_OpenDeviceByName(
            byref(self.handle), chDeviceName.encode("ascii")
        )

    # *******************************************************************************************************

    # *******************************************************************************************************
    # brief  通过序列号打开设备
    # param  self.handle                      [IN OUT]        相机句柄
    # param  chSerialNumber                   [IN]            序列号
    # return 成功，返回MV3D_RGBD_OK；错误，返回错误码

    # brief  open device by serial number
    # param  self.handle                      [IN OUT]        camera handle
    # param  chSerialNumber                   [IN]            serial number
    # return Success, return MV3D_RGBD_OK. Failure, return error code
    def MV3D_RGBD_OpenDeviceBySerialNumber(self, chSerialNumber):
        Mv3dRgbdDll.MV3D_RGBD_OpenDeviceBySerialNumber.argtype = (c_void_p, c_void_p)
        Mv3dRgbdDll.MV3D_RGBD_OpenDeviceBySerialNumber.restype = c_uint
        # C：MV3D_RGBD_API MV3D_RGBD_STATUS MV3D_RGBD_OpenDeviceBySerialNumber(HANDLE *handle, const char* chSerialNumber);
        return Mv3dRgbdDll.MV3D_RGBD_OpenDeviceBySerialNumber(
            byref(self.handle), chSerialNumber.encode("ascii")
        )

    # *******************************************************************************************************

    # *******************************************************************************************************
    # brief  通过IP打开设备,仅网口设备有效
    # param  self.handle                      [IN OUT]        相机句柄
    # param  chIP                             [IN]            IP地址，例如"10.114.71.116"
    # return 成功，返回MV3D_RGBD_OK；错误，返回错误码

    # brief  open device by ip，only network device is valid
    # param  self.handle                      [IN OUT]        camera handle
    # param  chIP                             [IN]            IP, for example, "10.114.71.116"
    # return Success, return MV3D_RGBD_OK. Failure, return error code
    def MV3D_RGBD_OpenDeviceByIp(self, chIP):
        Mv3dRgbdDll.MV3D_RGBD_OpenDeviceByIp.argtype = (c_void_p, c_void_p)
        Mv3dRgbdDll.MV3D_RGBD_OpenDeviceByIp.restype = c_uint
        # C：MV3D_RGBD_API MV3D_RGBD_STATUS MV3D_RGBD_OpenDeviceByIp(HANDLE *handle, const char* chIP);
        return Mv3dRgbdDll.MV3D_RGBD_OpenDeviceByIp(
            byref(self.handle), chIP.encode("ascii")
        )

    # *******************************************************************************************************

    # *******************************************************************************************************
    # param  self.handle                      [IN]            相机句柄
    # return 成功，返回MV3D_RGBD_OK；错误，返回错误码

    # param  self.handle                      [IN]            camera handle
    # return Success, return MV3D_RGBD_OK. Failure, return error code
    def MV3D_RGBD_CloseDevice(self):
        Mv3dRgbdDll.MV3D_RGBD_CloseDevice.argtype = c_void_p
        Mv3dRgbdDll.MV3D_RGBD_CloseDevice.restype = c_uint
        # C：MV3D_RGBD_API MV3D_RGBD_STATUS MV3D_RGBD_CloseDevice(HANDLE *handle);
        return Mv3dRgbdDll.MV3D_RGBD_CloseDevice(byref(self.handle))

    # *******************************************************************************************************

    # *******************************************************************************************************
    # brief  获取当前设备的详细信息
    # param  self.handle                      [IN]            相机句柄
    # param  pstDevInfo                       [IN][OUT]       返回给调用者有关相机设备信息结构体指针
    # return 成功,MV3D_RGBD_OK,失败,返回错误码

    # brief  get current device information
    # param  self.handle                      [IN]            camera handle
    # param  pstDevInfo                       [IN][OUT]       structure pointer of device information
    # return Success, return MV3D_RGBD_OK. Failure, return error code
    def MV3D_RGBD_GetDeviceInfo(self, pstDevInfo):
        Mv3dRgbdDll.MV3D_RGBD_GetDeviceInfo.argtype = (c_void_p, c_void_p)
        Mv3dRgbdDll.MV3D_RGBD_GetDeviceInfo.restype = c_uint
        # C：MV3D_RGBD_API MV3D_RGBD_STATUS MV3D_RGBD_GetDeviceInfo(HANDLE handle, MV3D_RGBD_DEVICE_INFO* pstDevInfo);
        return Mv3dRgbdDll.MV3D_RGBD_GetDeviceInfo(self.handle, pstDevInfo)

    # *******************************************************************************************************

    # *******************************************************************************************************
    # brief  配置IP,仅网口设备有效
    # param  chSerialNumber                   [IN]            序列号
    # param  pstIPConfig                      [IN]            IP配置，静态IP，DHCP等
    # return 成功,MV3D_RGBD_OK,失败,返回错误码

    # brief  ip configuration，only network device is valid
    # param  chSerialNumber                   [IN]            serial number
    # param  pstIPConfig                      [IN]            ip config, static ip，DHCP
    # return Success, return MV3D_RGBD_OK. Failure, return error code
    @staticmethod
    def MV3D_RGBD_SetIpConfig(chSerialNumber, pstIPConfig):
        Mv3dRgbdDll.MV3D_RGBD_SetIpConfig.argtype = (c_void_p, c_void_p)
        Mv3dRgbdDll.MV3D_RGBD_SetIpConfig.restype = c_uint
        # C：MV3D_RGBD_API MV3D_RGBD_STATUS MV3D_RGBD_SetIpConfig(const char* chSerialNumber, MV3D_RGBD_IP_CONFIG* pstIPConfig);
        return Mv3dRgbdDll.MV3D_RGBD_SetIpConfig(chSerialNumber, pstIPConfig)

    # *******************************************************************************************************

    # *******************************************************************************************************
    # brief  注册图像数据回调
    # param  self.handle                      [IN]            相机句柄
    # param  cbOutput                         [IN]            回调函数指针
    # param  pUser                            [IN]            用户自定义变量
    # return 成功，返回MV3D_RGBD_OK；错误，返回错误码

    # brief  register image data callback
    # param  self.handle                      [IN]            camera handle
    # param  cbOutput                         [IN]            callback function pointer
    # param  pUser                            [IN]            user defined variable
    # return Success, return MV3D_RGBD_OK. Failure, return error code
    def MV3D_RGBD_RegisterFrameCallBack(self, cbOutput, pUser):
        Mv3dRgbdDll.MV3D_RGBD_RegisterFrameCallBack.argtype = (
            c_void_p,
            c_void_p,
            c_void_p,
        )
        Mv3dRgbdDll.MV3D_RGBD_RegisterFrameCallBack.restype = c_uint
        # C：MV3D_RGBD_API MV3D_RGBD_STATUS MV3D_RGBD_RegisterFrameCallBack(HANDLE handle, MV3D_RGBD_FrameDataCallBack cbOutput, void* pUser);
        return Mv3dRgbdDll.MV3D_RGBD_RegisterFrameCallBack(self.handle, cbOutput, pUser)

    # *******************************************************************************************************

    # *******************************************************************************************************
    # brief  注册异常消息回调
    # param  self.handle：                    [IN]            相机句柄
    # param  cbException                      [IN]            异常回调函数指针
    # param  pUser                            [IN]            用户自定义变量
    # return 成功，返回MV3D_RGBD_OK；错误，返回错误码

    # brief  register exception message callBack
    # param  self.handle：                    [IN]            camera handle
    # param  cbException                      [IN]            exception message callBack function pointer
    # param  pUser                            [IN]            user defined variable
    # return Success, return MV3D_RGBD_OK. Failure, return error code
    def MV3D_RGBD_RegisterExceptionCallBack(self, cbException, pUser):
        Mv3dRgbdDll.MV3D_RGBD_RegisterExceptionCallBack.argtype = (
            c_void_p,
            c_void_p,
            c_void_p,
        )
        Mv3dRgbdDll.MV3D_RGBD_RegisterExceptionCallBack.restype = c_uint
        # C：MV3D_RGBD_API MV3D_RGBD_STATUS MV3D_RGBD_RegisterExceptionCallBack(HANDLE handle, MV3D_RGBD_ExceptionCallBack cbException, void* pUser);
        return Mv3dRgbdDll.MV3D_RGBD_RegisterExceptionCallBack(
            self.handle, cbException, pUser
        )

    # *******************************************************************************************************

    # *******************************************************************************************************
    # brief  开始取流前获取数据流配置列表
    # param  self.handle：                    [IN]            相机句柄
    # param  pstStreamCfgList                 [IN]            返回给调用者数据流配置列表指针
    # return 成功,MV3D_RGBD_OK,失败,返回错误码

    # brief  get stream cfg list before start working
    # param  self.handle：                    [IN]            camera handle
    # param  pstStreamCfgList                 [IN]            structure pointer of stream cfg list
    # return Success, return MV3D_RGBD_OK. Failure, return error code
    def MV3D_RGBD_GetStreamCfgList(self, pstStreamCfgList):
        Mv3dRgbdDll.MV3D_RGBD_GetStreamCfgList.argtype = (c_void_p, c_void_p)
        Mv3dRgbdDll.MV3D_RGBD_GetStreamCfgList.restype = c_uint
        # C：MV3D_RGBD_API MV3D_RGBD_STATUS MV3D_RGBD_GetStreamCfgList(HANDLE handle, MV3D_RGBD_STREAM_CFG_LIST* pstStreamCfgList);
        return Mv3dRgbdDll.MV3D_RGBD_GetStreamCfgList(self.handle, pstStreamCfgList)

    # *******************************************************************************************************

    # *******************************************************************************************************
    # brief  开始工作
    # param  self.handle                      [IN]            相机句柄
    # return 成功，返回MV3D_RGBD_OK；错误，返回错误码

    # brief  start  working
    # param  self.handle                      [IN]            camera handle
    # return Success, return MV3D_RGBD_OK. Failure, return error code
    def MV3D_RGBD_Start(self):
        Mv3dRgbdDll.MV3D_RGBD_Start.argtype = c_void_p
        Mv3dRgbdDll.MV3D_RGBD_Start.restype = c_uint
        # C：MV3D_RGBD_API MV3D_RGBD_STATUS MV3D_RGBD_Start(HANDLE handle);
        return Mv3dRgbdDll.MV3D_RGBD_Start(self.handle)

    # *******************************************************************************************************

    # *******************************************************************************************************
    # brief  停止工作
    # param  self.handle                      [IN]            相机句柄
    # return 成功，返回MV3D_RGBD_OK；错误，返回错误码

    # brief  stop working
    # param  self.handle                      [IN]            camera handle
    # return Success, return MV3D_RGBD_OK. Failure, return error code
    def MV3D_RGBD_Stop(self):
        Mv3dRgbdDll.MV3D_RGBD_Stop.argtype = c_void_p
        Mv3dRgbdDll.MV3D_RGBD_Stop.restype = c_uint
        # C：MV3D_RGBD_API MV3D_RGBD_STATUS MV3D_RGBD_Stop(HANDLE handle);
        return Mv3dRgbdDll.MV3D_RGBD_Stop(self.handle)

    # *******************************************************************************************************

    # *******************************************************************************************************
    # brief  轮询方式获取帧数据
    # param  self.handle                      [IN]            相机句柄
    # param  pstFrameData                     [IN][OUT]       数据指针
    # param  nTimeOut                         [IN]            超时时间（单位:毫秒）
    # return 成功，返回MV3D_RGBD_OK；错误，返回错误码

    # brief  fetch frame data
    # param  self.handle                      [IN]            camera handle
    # param  pstFrameData                     [IN][OUT]       data set pointer
    # param  nTimeOut                         [IN]            timevalue（Unit: ms）
    # return Success, return MV3D_RGBD_OK. Failure, return error code
    def MV3D_RGBD_FetchFrame(self, pstFrameData, nTimeOut):
        Mv3dRgbdDll.MV3D_RGBD_FetchFrame.argtype = (c_void_p, c_void_p, c_uint)
        Mv3dRgbdDll.MV3D_RGBD_FetchFrame.restype = c_uint
        # C：MV3D_RGBD_API MV3D_RGBD_STATUS MV3D_RGBD_FetchFrame(HANDLE handle, MV3D_RGBD_FRAME_DATA* pstFrameData, uint32_t nTimeOut);
        return Mv3dRgbdDll.MV3D_RGBD_FetchFrame(self.handle, pstFrameData, nTimeOut)

    # *******************************************************************************************************

    # *******************************************************************************************************
    # brief  执行设备软触发
    # param  self.handle                      [IN]            相机句柄
    # return 成功,返回MV3D_RGBD_OK,失败,返回错误码
    # remark 设置触发模式为打开，设置触发源为软触发并执行，软触发得到的图像帧包含触发id

    # brief  execute camera soft trigger
    # param  self.handle                      [IN]            camera handle
    # return Success, return MV3D_RGBD_OK. Failure, return error code
    # remark Set trigger mode to open, trigger source to software, and excute soft trigger. The image frame obtained by soft trigger contains the trigger id
    def MV3D_RGBD_SoftTrigger(self):
        Mv3dRgbdDll.MV3D_RGBD_SoftTrigger.argtype = c_void_p
        Mv3dRgbdDll.MV3D_RGBD_SoftTrigger.restype = c_uint
        # C：MV3D_RGBD_API MV3D_RGBD_STATUS MV3D_RGBD_SoftTrigger(HANDLE handle);
        return Mv3dRgbdDll.MV3D_RGBD_SoftTrigger(self.handle)

    # *******************************************************************************************************

    # *******************************************************************************************************
    # brief  执行带触发ID的设备软触发
    # param  self.handle                      [IN]            相机句柄
    # param  nTriggerId                       [IN]            触发ID
    # return 成功,返回MV3D_RGBD_OK,失败,返回错误码
    # remark 设置触发模式为打开，设置触发源为软触发并执行，软触发得到的图像帧包含触发id。
    # 通过对比输入的触发id参数与图像帧返回的触发id，判断当前帧是否有效

    # brief  execute camera soft trigger with trigger id
    # param  self.handle                      [IN]            camera handle
    # param  nTriggerId                       [IN]            trigger id
    # return Success, return MV3D_RGBD_OK. Failure, return error code
    # remark Set trigger mode to open, trigger source to software, and excute soft trigger. The image frame obtained by soft trigger contains the trigger id.
    # Compare the input trigger id parameter with the returned trigger id to determine whether the current frame is valid
    def MV3D_RGBD_SoftTriggerEx(self, nTriggerId):
        Mv3dRgbdDll.MV3D_RGBD_SoftTriggerEx.argtype = (c_void_p, c_uint)
        Mv3dRgbdDll.MV3D_RGBD_SoftTriggerEx.restype = c_uint
        # C：MV3D_RGBD_API MV3D_RGBD_STATUS MV3D_RGBD_SoftTriggerEx(HANDLE handle, uint32_t nTriggerId);
        return Mv3dRgbdDll.MV3D_RGBD_SoftTriggerEx(self.handle, nTriggerId)

    # *******************************************************************************************************

    # *******************************************************************************************************
    # brief  执行设备Command命令
    # param  self.handle                      [IN]            相机句柄
    # param  strKey                           [IN]            属性键值
    # return 成功,返回MV3D_RGBD_OK,失败,返回错误码

    # brief  execute camera command
    # param  self.handle                      [IN]            camera handle
    # param  strKey                           [IN]            key value
    # return Success, return MV3D_RGBD_OK. Failure, return error code
    def MV3D_RGBD_Execute(self, strKey):
        Mv3dRgbdDll.MV3D_RGBD_Execute.argtype = (c_void_p, c_void_p)
        Mv3dRgbdDll.MV3D_RGBD_Execute.restype = c_uint
        # C：MV3D_RGBD_API MV3D_RGBD_STATUS MV3D_RGBD_Execute(HANDLE handle, const char* strKey);
        return Mv3dRgbdDll.MV3D_RGBD_Execute(self.handle, strKey.encode("ascii"))

    # *******************************************************************************************************

    # *******************************************************************************************************
    # brief  获取相机当前标定信息
    # param  self.handle                      [IN]            相机句柄
    # param  nCoordinateType                  [IN]            坐标系类型，见Mv3dRgbdCoordinateType
    # param  pstCalibInfo                     [IN][OUT]       输出标定信息
    # return 成功,返回MV3D_RGBD_OK,失败,返回错误码

    # brief  get camera current calibration info
    # param  self.handle                      [IN]            camera handle
    # param  nCoordinateType                  [IN]            coordinate Type，refer to Mv3dRgbdCoordinateType
    # param  pstCalibInfo                     [IN][OUT]       calibration Info
    # return Success, return MV3D_RGBD_OK. Failure, return error code
    def MV3D_RGBD_GetCalibInfo(self, nCoordinateType, pstCalibInfo):
        Mv3dRgbdDll.MV3D_RGBD_GetCalibInfo.argtype = (c_void_p, c_uint, c_void_p)
        Mv3dRgbdDll.MV3D_RGBD_GetCalibInfo.restype = c_uint
        # C：MV3D_RGBD_API MV3D_RGBD_STATUS MV3D_RGBD_GetCalibInfo(HANDLE handle, uint32_t nCoordinateType, MV3D_RGBD_CALIB_INFO *pstCalibInfo);
        return Mv3dRgbdDll.MV3D_RGBD_GetCalibInfo(
            self.handle, nCoordinateType, pstCalibInfo
        )

    # *******************************************************************************************************

    # *******************************************************************************************************
    #  brief  获取相机内外参信息
    #  param  handle                          [IN]            相机句柄
    #  param  pstCameraParam                  [IN][OUT]       输出相机内/外参数信息
    #  return 成功,返回MV3D_RGBD_OK,失败,返回错误码

    #  brief  get camera intrinsic and extrinsic information
    #  param  handle                          [IN]            camera handle
    #  param  pstCameraParam                  [IN][OUT]       camera intrinsic and extrinsic Info
    def MV3D_RGBD_GetCameraParam(self, pstCameraParam):
        Mv3dRgbdDll.MV3D_RGBD_GetCameraParam.argtype = (c_void_p, c_void_p)
        Mv3dRgbdDll.MV3D_RGBD_GetCameraParam.restype = c_uint
        # C：MV3D_RGBD_API MV3D_RGBD_STATUS MV3D_RGBD_GetCameraParam(HANDLE handle, MV3D_RGBD_CAMERA_PARAM *pstCameraParam);
        return Mv3dRgbdDll.MV3D_RGBD_GetCameraParam(self.handle, pstCameraParam)

    # *******************************************************************************************************

    # *******************************************************************************************************
    # brief  设备升级
    # param  handle                            [IN]            相机句柄
    # param  pFilePathName                     [IN]            文件名
    # return 成功,返回MV3D_RGBD_OK,失败,返回错误码

    # brief  device upgrade
    # param  handle                            [IN]            camera handle
    # param  pFilePathName                     [IN]            file name
    # return Success, return MV3D_RGBD_OK. Failure, return error code
    def MV3D_RGBD_LocalUpgrade(self, pFilePathName):
        Mv3dRgbdDll.MV3D_RGBD_LocalUpgrade.argtype = (c_void_p, c_void_p)
        Mv3dRgbdDll.MV3D_RGBD_LocalUpgrade.restype = c_uint
        # C：MV3D_RGBD_API MV3D_RGBD_STATUS MV3D_RGBD_LocalUpgrade(HANDLE handle, const char* pFilePathName);
        return Mv3dRgbdDll.MV3D_RGBD_LocalUpgrade(
            self.handle, pFilePathName.encode("ascii")
        )

    # *******************************************************************************************************

    # *******************************************************************************************************
    # brief  获取升级进度
    # param  handle                            [IN]            相机句柄
    # param  pProcess                          [OUT]           进度接收地址
    # return 成功,返回MV3D_RGBD_OK,失败,返回错误码

    # brief  get upgrade progress
    # param  handle                            [IN]            camera handle
    # param  pProcess                          [OUT]           progress receiving address
    # return Success, return MV3D_RGBD_OK. Failure, return error code
    def MV3D_RGBD_GetUpgradeProcess(self, pProcess):
        Mv3dRgbdDll.MV3D_RGBD_GetUpgradeProcess.argtype = (c_void_p, c_void_p)
        Mv3dRgbdDll.MV3D_RGBD_GetUpgradeProcess.restype = c_uint
        # C：MV3D_RGBD_API MV3D_RGBD_STATUS MV3D_RGBD_GetUpgradeProcess(HANDLE handle, uint32_t* pProcess);
        return Mv3dRgbdDll.MV3D_RGBD_GetUpgradeProcess(self.handle, pProcess)

    # *******************************************************************************************************

    # *******************************************************************************************************
    # brief  获取相机参数值
    # param  handle                            [IN]            相机句柄
    # param  strKey                            [IN]            属性键值
    # param  pstParam                          [IN]            返回的相机参数结构体指针
    # return 成功,返回MV3D_RGBD_OK,失败,返回错误码

    # brief  get camera param value
    # param  handle                            [IN]            camera handle
    # param  strKey                            [IN]            key value
    # param  pstParam                          [IN]            structure pointer of camera param
    # return Success, return MV3D_RGBD_OK. Failure, return error code
    def MV3D_RGBD_GetParam(self, strkey, pstParam):
        Mv3dRgbdDll.MV3D_RGBD_GetParam.argtype = (c_void_p, c_void_p, c_void_p)
        Mv3dRgbdDll.MV3D_RGBD_GetParam.restype = c_uint
        # C：MV3D_RGBD_API MV3D_RGBD_STATUS MV3D_RGBD_GetParam(HANDLE handle, const char* strKey, MV3D_RGBD_PARAM* pstParam);
        return Mv3dRgbdDll.MV3D_RGBD_GetParam(
            self.handle, strkey.encode("ascii"), pstParam
        )

    # *******************************************************************************************************

    # *******************************************************************************************************
    # brief  设置相机参数值
    # param  handle                            [IN]            相机句柄
    # param  strKey                            [IN]            属性键值
    # param  pstParam                          [IN]            输入的相机参数结构体指针
    # return 成功,返回MV3D_RGBD_OK,失败,返回错误码

    # brief  set camera param value
    # param  handle                            [IN]            camera handle
    # param  strKey                            [IN]            key value
    # param  pstParam                          [IN]            structure pointer of camera param
    # return Success, return MV3D_RGBD_OK. Failure, return error code
    def MV3D_RGBD_SetParam(self, strkey, pstParam):
        Mv3dRgbdDll.MV3D_RGBD_SetParam.argtype = (c_void_p, c_void_p, c_void_p)
        Mv3dRgbdDll.MV3D_RGBD_SetParam.restype = c_uint
        # C：MV3D_RGBD_API MV3D_RGBD_STATUS MV3D_RGBD_SetParam(HANDLE handle, const char* strKey, MV3D_RGBD_PARAM* pstParam);
        return Mv3dRgbdDll.MV3D_RGBD_SetParam(
            self.handle, strkey.encode("ascii"), pstParam
        )

    # *******************************************************************************************************

    # *******************************************************************************************************
    # brief  导出相机参数
    # param  handle                            [IN]            相机句柄
    # param  pOutFileName                      [IN]            导出文件名称
    # return 成功，返回MV3D_RGBD_OK；错误，返回错误码

    # brief  export camera param
    # param  handle                            [IN]            camera handle
    # param  pOutFileName                      [IN]            export file name
    # return Success, return MV3D_RGBD_OK. Failure, return error code
    def MV3D_RGBD_ExportAllParam(self, pOutFileName):
        Mv3dRgbdDll.MV3D_RGBD_ExportAllParam.argtype = (c_void_p, c_void_p)
        Mv3dRgbdDll.MV3D_RGBD_ExportAllParam.restype = c_uint
        # C：MV3D_RGBD_API MV3D_RGBD_STATUS MV3D_RGBD_ExportAllParam(HANDLE handle, const char* pOutFileName);
        return Mv3dRgbdDll.MV3D_RGBD_ExportAllParam(
            self.handle, pOutFileName.encode("ascii")
        )

    # *******************************************************************************************************

    # *******************************************************************************************************
    # brief  导入相机参数
    # param  handle                            [IN]            相机句柄
    # param  pInFileName                       [IN]            导入文件名称
    # return 成功，返回MV3D_RGBD_OK；错误，返回错误码

    # brief  import camera param
    # param  handle                            [IN]            camera handle
    # param  pInFileName                       [IN]            import file name
    # return Success, return MV3D_RGBD_OK. Failure, return error code
    def MV3D_RGBD_ImportAllParam(self, pInFileName):
        Mv3dRgbdDll.MV3D_RGBD_ImportAllParam.argtype = (c_void_p, c_void_p)
        Mv3dRgbdDll.MV3D_RGBD_ImportAllParam.restype = c_uint
        # C：MV3D_RGBD_API MV3D_RGBD_STATUS MV3D_RGBD_ImportAllParam(HANDLE handle, const char* pInFileName);
        return Mv3dRgbdDll.MV3D_RGBD_ImportAllParam(
            self.handle, pInFileName.encode("ascii")
        )

    # *******************************************************************************************************

    # *******************************************************************************************************
    # brief  从相机读取文件
    # param  handle                            [IN]            相机句柄
    # param  pstFileAccess                     [IN]            文件存取结构体
    # return 成功，返回MV3D_RGBD_OK；错误，返回错误码

    # brief  read the file from the camera
    # param  handle                            [IN]            camera handle
    # param  pstFileAccess                     [IN]            file access structure
    # return Success, return MV3D_RGBD_OK. Failure, return error code
    def MV3D_RGBD_FileAccessRead(self, pstFileAccess):
        Mv3dRgbdDll.MV3D_RGBD_FileAccessRead.argtype = (c_void_p, c_void_p)
        Mv3dRgbdDll.MV3D_RGBD_FileAccessRead.restype = c_uint
        # C：MV3D_RGBD_API MV3D_RGBD_STATUS MV3D_RGBD_FileAccessRead(void* handle, MV3D_RGBD_FILE_ACCESS * pstFileAccess);
        return Mv3dRgbdDll.MV3D_RGBD_FileAccessRead(self.handle, pstFileAccess)

    # *******************************************************************************************************

    # *******************************************************************************************************
    # brief  将文件写入相机
    # param  handle                            [IN]            相机句柄
    # param  pstFileAccess                     [IN]            文件存取结构体
    # return 成功，返回MV3D_RGBD_OK；错误，返回错误码

    # brief  write the file to camera
    # param  handle                            [IN]            camera handle
    # param  pstFileAccess                     [IN]            file access structure
    # return Success, return MV3D_RGBD_OK. Failure, return error code
    def MV3D_RGBD_FileAccessWrite(self, pstFileAccess):
        Mv3dRgbdDll.MV3D_RGBD_FileAccessWrite.argtype = (c_void_p, c_void_p)
        Mv3dRgbdDll.MV3D_RGBD_FileAccessWrite.restype = c_uint
        # C：MV3D_RGBD_API MV3D_RGBD_STATUS MV3D_RGBD_FileAccessWrite(void* handle, MV3D_RGBD_FILE_ACCESS * pstFileAccess);
        return Mv3dRgbdDll.MV3D_RGBD_FileAccessWrite(self.handle, pstFileAccess)

    # *******************************************************************************************************

    # *******************************************************************************************************
    # brief  获取文件存取的进度
    # param  handle                            [IN]            相机句柄
    # param  pstFileAccessProgress             [IN]            文件存取进度结构体
    # return 成功，返回MV3D_RGBD_OK；错误，返回错误码

    # brief  get file access progress
    # param  handle                            [IN]            camera handle
    # param  pstFileAccessProgress             [IN]            file access progress structure
    # return Success, return MV3D_RGBD_OK. Failure, return error code
    def MV3D_RGBD_GetFileAccessProgress(self, pstFileAccessProgress):
        Mv3dRgbdDll.MV3D_RGBD_GetFileAccessProgress.argtype = (c_void_p, c_void_p)
        Mv3dRgbdDll.MV3D_RGBD_GetFileAccessProgress.restype = c_uint
        # C：MV3D_RGBD_API MV3D_RGBD_STATUS MV3D_RGBD_GetFileAccessProgress(void* handle, MV3D_RGBD_FILE_ACCESS_PROGRESS * pstFileAccessProgress);
        return Mv3dRgbdDll.MV3D_RGBD_GetFileAccessProgress(
            self.handle, pstFileAccessProgress
        )

    # *******************************************************************************************************

    # *******************************************************************************************************
    # brief  检查相机功能支持情况
    # param  handle                            [IN]            相机句柄
    # param  enFuncType                        [IN]            功能类型
    # return 成功，返回MV3D_RGBD_OK；错误，返回错误码
    # remark 调用该接口检查相机是否支持某项功能，例如检查相机是否支持触发id功能，如果支持，则后续可以调用MV3D_RGBD_SoftTriggerEx接口，执行带触发id的软触发

    # brief  check camera function support
    # param  handle                            [IN]            camera handle
    # param  enFuncType                        [IN]            function type
    # return Success, return MV3D_RGBD_OK. Failure, return error code
    # remark Call this interface to check whether the camera supports a certain function, for example, check whether the camera supports the trigger id function.
    # If camera supports the trigger id function, you can call the MV3D_RGBD_SoftTriggerEx interface to execute the soft trigger with the trigger id
    def MV3D_RGBD_CheckCameraFuncSupport(self, enFuncType):
        Mv3dRgbdDll.MV3D_RGBD_CheckCameraFuncSupport.argtype = (c_void_p, c_uint)
        Mv3dRgbdDll.MV3D_RGBD_CheckCameraFuncSupport.restype = c_uint
        # C：MV3D_RGBD_API MV3D_RGBD_STATUS MV3D_RGBD_CheckCameraFuncSupport(void* handle, Mv3dRgbdFunctionType enFuncType);
        return Mv3dRgbdDll.MV3D_RGBD_CheckCameraFuncSupport(self.handle, enFuncType)

    # *******************************************************************************************************

    # *******************************************************************************************************
    # brief  清除数据缓存
    # param  handle                            [IN]            相机句柄
    # return 成功，返回MV3D_RGBD_OK；错误，返回错误码

    # brief  clear data buffer
    # param  handle                            [IN]            camera handle
    # return Success, return MV3D_RGBD_OK. Failure, return error code
    def MV3D_RGBD_ClearDataBuffer(self):
        Mv3dRgbdDll.MV3D_RGBD_ClearDataBuffer.argtype = c_void_p
        Mv3dRgbdDll.MV3D_RGBD_ClearDataBuffer.restype = c_uint
        # C：MV3D_RGBD_API MV3D_RGBD_STATUS MV3D_RGBD_ClearDataBuffer(void* handle);
        return Mv3dRgbdDll.MV3D_RGBD_ClearDataBuffer(self.handle)

    # *******************************************************************************************************

    # *******************************************************************************************************
    # brief  RGBD相机深度图像转换点云图像
    # param  handle                            [IN]            相机句柄
    # param  pstDepthImage                     [IN]            深度图数据
    # param  pstPointCloudImage                [OUT]           点云图数据
    # return 成功，返回MV3D_RGBD_OK；错误，返回错误码

    # brief  depth image convert to pointcloud image
    # param  handle                            [IN]            camera handle
    # param  pstDepthImage                     [IN]            depth  data
    # param  pstPointCloudImage                [OUT]           point cloud data
    # return Success, return MV3D_RGBD_OK. Failure,return error code
    def MV3D_RGBD_MapDepthToPointCloud(self, pstDepthImage, pstPointCloudImage):
        Mv3dRgbdDll.MV3D_RGBD_MapDepthToPointCloud.argtype = (
            c_void_p,
            c_void_p,
            c_void_p,
        )
        Mv3dRgbdDll.MV3D_RGBD_MapDepthToPointCloud.restype = c_uint
        # C：MV3D_RGBD_API MV3D_RGBD_STATUS  MV3D_RGBD_MapDepthToPointCloud(void* handle, MV3D_RGBD_IMAGE_DATA* pstDepthImage, MV3D_RGBD_IMAGE_DATA* pstPointCloudImage);
        return Mv3dRgbdDll.MV3D_RGBD_MapDepthToPointCloud(
            self.handle, pstDepthImage, pstPointCloudImage
        )

    # *******************************************************************************************************

    # *******************************************************************************************************
    # brief  RGBD相机深度图像转换点云图像（无句柄）
    # param  pstDepthImage                     [IN]            深度图数据
    # param  pstCalibInfo                      [IN]            标定信息
    # param  fZunit                            [IN]            深度图量纲(mm)
    # param  pstPointCloudImage                [OUT]           点云图数据
    # return 成功，返回MV3D_RGBD_OK；错误，返回错误码

    # brief  depth image convert to pointcloud image without handle
    # param  pstDepthImage                     [IN]            depth  data
    # param  pstCalibInfo                      [IN]            calib info
    # param  fZunit                            [IN]            dimension(mm)
    # param  pstPointCloudImage                [OUT]           point cloud data
    # return Success, return MV3D_RGBD_OK. Failure,return error code
    @staticmethod
    def MV3D_RGBD_MapDepthToPointCloudEx(
        pstDepthImage, pstCalibInfo, fZunit, pstPointCloudImage
    ):
        Mv3dRgbdDll.MV3D_RGBD_MapDepthToPointCloudEx.argtype = (
            c_void_p,
            c_void_p,
            c_float,
            c_void_p,
        )
        Mv3dRgbdDll.MV3D_RGBD_MapDepthToPointCloudEx.restype = c_uint
        # C：MV3D_RGBD_API MV3D_RGBD_STATUS MV3D_RGBD_MapDepthToPointCloudEx(MV3D_RGBD_IMAGE_DATA* pstDepthImage, MV3D_RGBD_CALIB_INFO* pstCalibInfo, float fZunit, MV3D_RGBD_IMAGE_DATA* pstPointCloudImage);
        return Mv3dRgbdDll.MV3D_RGBD_MapDepthToPointCloudEx(
            pstDepthImage, pstCalibInfo, ctypes.c_float(fZunit), pstPointCloudImage
        )

    # *******************************************************************************************************

    # *******************************************************************************************************
    # brief  图像坐标系转换
    # param  pstInImage                        [IN]            输入图像数据
    # param  fZunit                            [IN]            深度图量纲(mm)
    # param  pstOutImage                       [OUT]           输出图像数据
    # param  pstCameraParam                    [IN][OUT]       相机内/外参数信息
    # return 成功，返回MV3D_RGBD_OK；错误，返回错误码

    # brief  image convert coordinate to rgb coordinate
    # param  pstInImage                        [IN]            input image data
    # param  fZunit                            [IN]            dimension(mm)
    # param  pstOutImage                       [OUT]           output image data
    # param  pstCameraParam                    [IN][OUT]       camera intrinsic and extrinsic Info
    # return Success, return MV3D_RGBD_OK. Failure,return error code
    @staticmethod
    def MV3D_RGBD_ImageCoordinateTrans(pstInImage, fZunit, pstOutImage, pstCameraParam):
        Mv3dRgbdDll.MV3D_RGBD_ImageCoordinateTrans.argtype = (
            c_void_p,
            c_float,
            c_void_p,
            c_void_p,
        )
        Mv3dRgbdDll.MV3D_RGBD_ImageCoordinateTrans.restype = c_uint
        # C：MV3D_RGBD_API MV3D_RGBD_STATUS MV3D_RGBD_ImageCoordinateTrans(MV3D_RGBD_IMAGE_DATA* pstInImage, float fZunit, MV3D_RGBD_IMAGE_DATA* pstOutImage, MV3D_RGBD_CAMERA_PARAM* pstCameraParam);
        return Mv3dRgbdDll.MV3D_RGBD_ImageCoordinateTrans(
            pstInImage, ctypes.c_float(fZunit), pstOutImage, pstCameraParam
        )

    # *******************************************************************************************************

    # *******************************************************************************************************
    # brief  深度图,RGB图和原始图存图接口
    #       深度图格式:C16
    #       RGB图格式:RGB8_Planar/YUV422/YUV420SP_NV12/YUV420SP_NV21
    #       原始图格式:Mono8(仅支持bmp格式)
    # param  handle                            [IN]            相机句柄
    # param  pstImage                          [IN]            图像数据
    # param  enFileType                        [IN]            文件类型
    # param  chFileName                        [IN]            文件名称
    # return 成功，返回MV3D_RGBD_OK；错误，返回错误码

    # brief  depth and rgb image save image to file
    #       depth image format: C16
    #       rgb image format: RGB8_Planar/YUV422/YUV420SP_NV12/YUV420SP_NV21
    #       mono image format: Mono8(only support bmp file type)
    # param  handle                            [IN]            camera handle
    # param  pstImage                          [IN]            image data
    # param  enFileType                        [IN]            file type
    # param  chFileName                        [IN]            file name
    # return Success, return MV3D_RGBD_OK. Failure, return error code
    def MV3D_RGBD_SaveImage(self, pstImage, enFileType, chFileName):
        Mv3dRgbdDll.MV3D_RGBD_SaveImage.argtype = (c_void_p, c_void_p, c_int, c_void_p)
        Mv3dRgbdDll.MV3D_RGBD_SaveImage.restype = c_uint
        # C：MV3D_RGBD_API MV3D_RGBD_STATUS MV3D_RGBD_SaveImage(void* handle, MV3D_RGBD_IMAGE_DATA* pstImage, Mv3dRgbdFileType enFileType, const char* chFileName);
        return Mv3dRgbdDll.MV3D_RGBD_SaveImage(
            self.handle, pstImage, enFileType, chFileName.encode("ascii")
        )

    # *******************************************************************************************************

    # *******************************************************************************************************
    # brief  点云图存图接口
    # param  handle                            [IN]            相机句柄
    # param  pstImage                          [IN]            图像数据
    # param  enPointCloudFileType              [IN]            点云图文件类型
    # param  chFileName                        [IN]            文件名称
    # return 成功，返回MV3D_RGBD_OK；错误，返回错误码

    # brief  pointcloud image save image to file
    # param  handle                            [IN]            camera handle
    # param  pstImage                          [IN]            image data
    # param  enPointCloudFileType              [IN]            pointcloud image file type
    # param  chFileName                        [IN]            file name
    # return Success, return MV3D_RGBD_OK. Failure, return error code
    def MV3D_RGBD_SavePointCloudImage(self, pstImage, enPointCloudFileType, chFileName):
        Mv3dRgbdDll.MV3D_RGBD_SavePointCloudImage.argtype = (
            c_void_p,
            c_void_p,
            c_int,
            c_void_p,
        )
        Mv3dRgbdDll.MV3D_RGBD_SavePointCloudImage.restype = c_uint
        # C：MV3D_RGBD_API MV3D_RGBD_STATUS MV3D_RGBD_SavePointCloudImage(void* handle, MV3D_RGBD_IMAGE_DATA* pstImage, Mv3dRgbdPointCloudFileType enPointCloudFileType, const char* chFileName);
        return Mv3dRgbdDll.MV3D_RGBD_SavePointCloudImage(
            self.handle, pstImage, enPointCloudFileType, chFileName.encode("ascii")
        )

    # *******************************************************************************************************

    # *******************************************************************************************************
    # brief  纹理点云存图接口
    #       纹理图格式: RGB8_Planar/YUV422/YUV420SP_NV12/YUV420SP_NV21
    #       保存的点云图格式: PLY_ASCII/PLC_Binary/PCD_ASCII/PCD_Binary
    # param  handle                            [IN]            相机句柄
    # param  pstPointCloudImage                [IN]            点云图像数据
    # param  pstTexture                        [IN]            图像纹理数据
    # param  enPointCloudFileType              [IN]            点云图文件类型
    # param  chFileName                        [IN]            文件名称
    # return 成功，返回MV3D_RGBD_OK；错误，返回错误码

    # brief  textured pointcloud image save image to file
    #       texture image format:RGB8_Planar/YUV422/YUV420SP_NV12/YUV420SP_NV21
    #       point cloud file type: PLY_ASCII/PLC_Binary/PCD_ASCII/PCD_Binary
    # param  handle                            [IN]            camera handle
    # param  pstPointCloudImage                [IN]            pointcloude image data
    # param  pstTexture                        [IN]            image texture data
    # param  enPointCloudFileType              [IN]            pointcloud image file type
    # param  chFileName                        [IN]            file name
    # return Success, return MV3D_RGBD_OK. Failure, return error code
    def MV3D_RGBD_SaveTexturedPointCloudImage(
        self, pstPointCloudImage, pstTexture, enPointCloudFileType, chFileName
    ):
        Mv3dRgbdDll.MV3D_RGBD_SaveTexturedPointCloudImage.argtype = (
            c_void_p,
            c_void_p,
            c_void_p,
            c_int,
            c_void_p,
        )
        Mv3dRgbdDll.MV3D_RGBD_SaveTexturedPointCloudImage.restype = c_uint
        # C：MV3D_RGBD_API MV3D_RGBD_STATUS MV3D_RGBD_SaveTexturedPointCloudImage(void* handle, MV3D_RGBD_IMAGE_DATA* pstPointCloudImage, MV3D_RGBD_IMAGE_DATA* pstTexture, Mv3dRgbdPointCloudFileType enPointCloudFileType, const char* chFileName);
        return Mv3dRgbdDll.MV3D_RGBD_SaveTexturedPointCloudImage(
            self.handle,
            pstPointCloudImage,
            pstTexture,
            enPointCloudFileType,
            chFileName.encode("ascii"),
        )

    # *******************************************************************************************************

    # *******************************************************************************************************
    # brief  显示深度和RGB图像接口
    #       深度图格式:C16
    #       RGB图格式:RGB8_Planar/YUV422/YUV420SP_NV12/YUV420SP_NV21
    # param  handle                            [IN]            相机句柄
    # param  pstImage                          [IN]            图像数据
    # param  hWnd                              [IN]            窗口句柄
    # return 成功，返回MV3D_RGBD_OK；错误，返回错误码

    # brief  display depth and rgb image
    #       depth image format: C16
    #       rgb image format: RGB8_Planar/YUV422/YUV420SP_NV12/YUV420SP_NV21
    # param  handle                            [IN]            camera handle
    # param  pstImage                          [IN]            image data
    # param  hWnd                              [IN]            windows handle
    # return Success, return MV3D_RGBD_OK. Failure, return error code
    def MV3D_RGBD_DisplayImage(self, pstImage, hWnd):
        Mv3dRgbdDll.MV3D_RGBD_DisplayImage.argtype = (c_void_p, c_void_p, c_void_p)
        Mv3dRgbdDll.MV3D_RGBD_DisplayImage.restype = c_uint
        # C:MV3D_RGBD_API MV3D_RGBD_STATUS MV3D_RGBD_DisplayImage(void* handle, MV3D_RGBD_IMAGE_DATA* pstImage, void* hWnd)
        return Mv3dRgbdDll.MV3D_RGBD_DisplayImage(self.handle, pstImage, hWnd)

    # *******************************************************************************************************

    # *******************************************************************************************************
    # brief  深度图转伪彩图接口
    #       深度图格式:C16
    #       伪彩图格式:RGB8_Planar
    # param  pstDepthImage                     [IN]            深度图像数据
    # param  pstConvertParam                   [IN]            深度图转伪彩配置参数
    # param  pstColorImage                     [OUT]           伪彩图像数据
    # return 成功，返回MV3D_RGBD_OK；错误，返回错误码

    # brief  display depth and rgb image
    #       depth image format: C16
    #       rgb image format: RGB8_Planar
    # param  pstDepthImage                     [IN]            depth image data
    # param  pstConvertParam                   [IN]            depth convert to color image parameters
    # param  pstColorImage                     [OUT]           color image data
    # return Success, return MV3D_RGBD_OK. Failure, return error code
    @staticmethod
    def MV3D_RGBD_MapDepthToColor(pstDepthImage, pstConvertParam, pstColorImage):
        Mv3dRgbdDll.MV3D_RGBD_MapDepthToColor.argtype = (c_void_p, c_void_p, c_void_p)
        Mv3dRgbdDll.MV3D_RGBD_MapDepthToColor.restype = c_uint
        # C:MV3D_RGBD_API MV3D_RGBD_STATUS MV3D_RGBD_MapDepthToColor(MV3D_RGBD_IMAGE_DATA* pstDepthImage, MV3D_RGBD_CONVERT_COLOR_PAPRAM* pstConvertParam, MV3D_RGBD_IMAGE_DATA* pstColorImage)
        return Mv3dRgbdDll.MV3D_RGBD_MapDepthToColor(
            pstDepthImage, pstConvertParam, pstColorImage
        )

    # *******************************************************************************************************

    # *******************************************************************************************************
    # brief  点云转UV坐标接口
    # param  pstCloudPointImage                [IN]            点云图像数据
    # param  pstCalibInfo                      [IN]            标定参数
    # param  pstUVMap                          [OUT]           UV坐标数据
    # return 成功，返回MV3D_RGBD_OK；错误，返回错误码

    # brief  convert point cloud to uv coordinate
    # param  pstCloudPointImage                [IN]            point cloud image data
    # param  pstCalibInfo                      [IN]            calib info
    # param  pstUVMap                          [OUT]           uv coordinate data
    # return Success, return MV3D_RGBD_OK. Failure, return error code
    @staticmethod
    def MV3D_RGBD_MapPointCloudToUV(pstCloudPointImage, pstCalibInfo, pstUvMap):
        Mv3dRgbdDll.MV3D_RGBD_MapPointCloudToUV.argtype = (c_void_p, c_void_p, c_void_p)
        Mv3dRgbdDll.MV3D_RGBD_MapPointCloudToUV.restype = c_uint
        # C:MV3D_RGBD_API MV3D_RGBD_STATUS MV3D_RGBD_MapPointCloudToUV(MV3D_RGBD_IMAGE_DATA* pstCloudPointImage, MV3D_RGBD_CALIB_INFO* pstCalibInfo, MV3D_RGBD_UV_DATA* pstUvMap)
        return Mv3dRgbdDll.MV3D_RGBD_MapPointCloudToUV(
            pstCloudPointImage, pstCalibInfo, pstUvMap
        )

    # *******************************************************************************************************
