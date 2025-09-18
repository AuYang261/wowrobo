# -- coding: utf-8 --
# Description: 连接相机获取RGB+Depth图像demo
import threading
import msvcrt
import ctypes
import time
import os
import struct
from ctypes import *
import tkinter as tk
from Mv3dRgbdImport.Mv3dRgbdDefine import *
from Mv3dRgbdImport.Mv3dRgbdApi import *
from Mv3dRgbdImport.Mv3dRgbdDefine import (
    DeviceType_Ethernet,
    DeviceType_USB,
    DeviceType_Ethernet_Vir,
    DeviceType_USB_Vir,
    MV3D_RGBD_FLOAT_EXPOSURETIME,
    ParamType_Float,
    ParamType_Int,
    ParamType_Enum,
    CoordinateType_Depth,
    MV3D_RGBD_FLOAT_Z_UNIT,
    MV3D_RGBD_OK,
    ImageType_Depth,
    ImageType_RGB8_Planar,
    ImageType_YUV422,
    ImageType_YUV420SP_NV12,
    ImageType_YUV420SP_NV21,
    MV3D_RGBD_CAMERA_PARAM,
    MV3D_RGBD_INT_IMAGEALIGN,
    FileType_BMP,
)
import numpy as np
import cv2


def PrintCameraParam(stCameraParam):
    print("-------- Depth Calib Info --------")
    print("Depth Width: %d" % stCameraParam.stDepthCalibInfo.nWidth)
    print("Depth Height: %d" % stCameraParam.stDepthCalibInfo.nHeight)
    print("--------")
    for i in range(0, 9):
        print(
            "Depth Intrinsic[%d]: %.10f"
            % (i, stCameraParam.stDepthCalibInfo.stIntrinsic.fData[i])
        )
    print("--------")
    for i in range(0, 12):
        print(
            "Depth Distortion[%d]: %.10f"
            % (i, stCameraParam.stDepthCalibInfo.stDistortion.fData[i])
        )

    print("-------- Rgb Calib Info --------")
    print("Rgb Width: %d" % stCameraParam.stRgbCalibInfo.nWidth)
    print("Rgb Height: %d" % stCameraParam.stRgbCalibInfo.nHeight)
    print("--------")
    for i in range(0, 9):
        print(
            "Rgb Intrinsic[%d]: %.10f"
            % (i, stCameraParam.stRgbCalibInfo.stIntrinsic.fData[i])
        )
    print("--------")
    for i in range(0, 12):
        print(
            "Rgb Distortion[%d]: %.10f"
            % (i, stCameraParam.stRgbCalibInfo.stDistortion.fData[i])
        )
    print("-------- Depth to Rgb Extrinsic --------")
    for i in range(0, 16):
        print(
            "Depth to Rgb Extrinsic[%d]: %.10f"
            % (i, stCameraParam.stDepth2RgbExtrinsic.fData[i])
        )
    print("--------")
    return


def parse_frame_data(stImageData) -> list[bool, np.ndarray]:
    if ImageType_Depth == stImageData.enImageType:
        strMode = string_at(
            stImageData.pData,
            stImageData.nDataLen,
        )
        image = np.frombuffer(strMode, dtype=np.uint16)
        image = image.reshape(
            stImageData.nHeight,
            stImageData.nWidth,
        )
        # normalize to 0-255
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return True, image
    elif ImageType_RGB8_Planar == stImageData.enImageType:
        strMode = string_at(
            stImageData.pData,
            stImageData.nDataLen,
        )
        image = np.frombuffer(strMode, dtype=np.uint8)
        image = image.reshape(
            3,
            stImageData.nHeight * stImageData.nWidth,
        )
        image = image.T.reshape(stImageData.nDataLen)
        image = image.reshape(
            stImageData.nHeight,
            stImageData.nWidth,
            3,
        )
    elif ImageType_YUV422 == stImageData.enImageType:
        YUV = string_at(
            stImageData.pData,
            stImageData.nDataLen,
        )
        YUV_np = np.frombuffer(YUV, dtype=np.uint8)
        YUV_np = YUV_np.reshape(
            (int)(stImageData.nHeight * stImageData.nWidth / 2),
            4,
        )
        YUV_np = YUV_np.T.reshape(stImageData.nDataLen)
        Y1 = YUV_np[0 : (int)(stImageData.nHeight * stImageData.nWidth / 2)]
        U = YUV_np[
            (int)(stImageData.nHeight * stImageData.nWidth / 2) : stImageData.nHeight
            * stImageData.nWidth
        ]
        U = np.repeat(U, 2)
        Y2 = YUV_np[
            stImageData.nHeight
            * stImageData.nWidth : (int)(
                3 * stImageData.nHeight * stImageData.nWidth / 2
            )
        ]
        V = YUV_np[
            (int)(3 * stImageData.nHeight * stImageData.nWidth / 2) : 2
            * stImageData.nHeight
            * stImageData.nWidth
        ]
        V = np.repeat(V, 2)

        Y = np.append(Y1, Y2)
        Y = Y.reshape(
            2,
            (int)(stImageData.nHeight * stImageData.nWidth / 2),
        ).T
        Y = Y.reshape(stImageData.nHeight * stImageData.nWidth)

        R = Y + 1.370705 * (V - np.array([128]))
        G = Y - 0.698001 * (U - np.array([128])) - (0.703125 * (V - np.array([128])))
        B = Y + 1.732446 * (U - np.array([128]))

        R = np.where(R < 0, 0, R)
        R = np.where(R > 255, 255, R)
        G = np.where(G < 0, 0, G)
        G = np.where(G > 255, 255, G)
        B = np.where(B < 0, 0, B)
        B = np.where(B > 255, 255, B)

        RGB = np.zeros(
            (
                stImageData.nHeight,
                stImageData.nWidth,
                3,
            ),
            dtype=np.uint8,
        )
        RGB[:, :, 2] = B.reshape(
            stImageData.nHeight,
            stImageData.nWidth,
        )
        RGB[:, :, 1] = G.reshape(
            stImageData.nHeight,
            stImageData.nWidth,
        )
        RGB[:, :, 0] = R.reshape(
            stImageData.nHeight,
            stImageData.nWidth,
        )
        image = RGB[:, :, ::-1]
    elif ImageType_YUV420SP_NV12 == stImageData.enImageType:
        YUV = string_at(
            stImageData.pData,
            stImageData.nDataLen,
        )
        YUV_np = np.frombuffer(YUV, dtype=np.uint8)
        Y = YUV_np[0 : stImageData.nHeight * stImageData.nWidth]
        UV = YUV_np[
            stImageData.nHeight
            * stImageData.nWidth : (int)(
                3 * stImageData.nHeight * stImageData.nWidth / 2
            )
        ]
        UV = UV.reshape(
            (int)(stImageData.nHeight * stImageData.nWidth / 4),
            2,
        )
        UV = UV.T.reshape((int)(stImageData.nHeight * stImageData.nWidth / 2))
        U = UV[0 : (int)(stImageData.nHeight * stImageData.nWidth / 4)]
        U = np.repeat(U, 2).reshape(
            (int)(stImageData.nHeight / 2),
            stImageData.nWidth,
        )
        U = np.repeat(U, 2, axis=0).reshape(stImageData.nHeight * stImageData.nWidth)

        V = UV[
            (int)(stImageData.nHeight * stImageData.nWidth / 4) : (int)(
                stImageData.nHeight * stImageData.nWidth / 2
            )
        ]
        V = np.repeat(V, 2).reshape(
            (int)(stImageData.nHeight / 2),
            stImageData.nWidth,
        )
        V = np.repeat(V, 2, axis=0).reshape(stImageData.nHeight * stImageData.nWidth)

        R = Y + (140 * (V - np.array([128]))) / 100
        G = Y - (34 * (U - np.array([128]))) / 100 - (71 * (V - np.array([128]))) / 100
        B = Y + (177 * (U - np.array([128]))) / 100

        R = np.where(R < 0, 0, R)
        R = np.where(R > 255, 255, R)
        G = np.where(G < 0, 0, G)
        G = np.where(G > 255, 255, G)
        B = np.where(B < 0, 0, B)
        B = np.where(B > 255, 255, B)

        RGB = np.zeros(
            (
                stImageData.nHeight,
                stImageData.nWidth,
                3,
            ),
            dtype=np.uint8,
        )
        RGB[:, :, 2] = B.reshape(
            stImageData.nHeight,
            stImageData.nWidth,
        )
        RGB[:, :, 1] = G.reshape(
            stImageData.nHeight,
            stImageData.nWidth,
        )
        RGB[:, :, 0] = R.reshape(
            stImageData.nHeight,
            stImageData.nWidth,
        )
        image = RGB
    elif ImageType_YUV420SP_NV21 == stImageData.enImageType:
        YUV = string_at(
            stImageData.pData,
            stImageData.nDataLen,
        )
        YUV_np = np.frombuffer(YUV, dtype=np.uint8)
        Y = YUV_np[0 : stImageData.nHeight * stImageData.nWidth]
        UV = YUV_np[
            stImageData.nHeight
            * stImageData.nWidth : (int)(
                3 * stImageData.nHeight * stImageData.nWidth / 2
            )
        ]
        UV = UV.reshape(
            (int)(stImageData.nHeight * stImageData.nWidth / 4),
            2,
        )
        UV = UV.T.reshape((int)(stImageData.nHeight * stImageData.nWidth / 2))
        V = UV[0 : (int)(stImageData.nHeight * stImageData.nWidth / 4)]
        V = np.repeat(V, 2).reshape(
            (int)(stImageData.nHeight / 2),
            stImageData.nWidth,
        )
        V = np.repeat(V, 2, axis=0).reshape(stImageData.nHeight * stImageData.nWidth)

        U = UV[
            (int)(stImageData.nHeight * stImageData.nWidth / 4) : (int)(
                stImageData.nHeight * stImageData.nWidth / 2
            )
        ]
        U = np.repeat(U, 2).reshape(
            (int)(stImageData.nHeight / 2),
            stImageData.nWidth,
        )
        U = np.repeat(U, 2, axis=0).reshape(stImageData.nHeight * stImageData.nWidth)

        R = Y + (140 * (V - np.array([128]))) / 100
        G = Y - (34 * (U - np.array([128]))) / 100 - (71 * (V - np.array([128]))) / 100
        B = Y + (177 * (U - np.array([128]))) / 100

        R = np.where(R < 0, 0, R)
        R = np.where(R > 255, 255, R)
        G = np.where(G < 0, 0, G)
        G = np.where(G > 255, 255, G)
        B = np.where(B < 0, 0, B)
        B = np.where(B > 255, 255, B)

        RGB = np.zeros(
            (
                stImageData.nHeight,
                stImageData.nWidth,
                3,
            ),
            dtype=np.uint8,
        )
        RGB[:, :, 2] = B.reshape(
            stImageData.nHeight,
            stImageData.nWidth,
        )
        RGB[:, :, 1] = G.reshape(
            stImageData.nHeight,
            stImageData.nWidth,
        )
        RGB[:, :, 0] = R.reshape(
            stImageData.nHeight,
            stImageData.nWidth,
        )
        image = RGB
    else:
        print("not support this format")
        return False, None
    return False, image


def work_thread(camera=0):
    while True:
        stFrameData = MV3D_RGBD_FRAME_DATA()
        ret = camera.MV3D_RGBD_FetchFrame(pointer(stFrameData), 5000)
        if ret == 0:
            for i in range(0, stFrameData.nImageCount):
                print(
                    "MV3D_RGBD_FetchFrame[%d]:nFrameNum[%d],nDataLen[%d],nWidth[%d],nHeight[%d],ImageType[%d]"
                    % (
                        i,
                        stFrameData.stImageData[i].nFrameNum,
                        stFrameData.stImageData[i].nDataLen,
                        stFrameData.stImageData[i].nWidth,
                        stFrameData.stImageData[i].nHeight,
                        stFrameData.stImageData[i].enImageType,
                    )
                )
                depth, image = parse_frame_data(stFrameData.stImageData[i])
                if depth:
                    cv2.imwrite("Depth.png", image)
                else:
                    cv2.imwrite("RGB.png", image)
                    # 锐化
                    kernel = np.array(
                        [[-1, -1.5, -1], [-1.5, 11, -1.5], [-1, -1.5, -1]],
                        np.float32,
                    )
                    image = cv2.filter2D(image, -1, kernel=kernel)
                    cv2.imwrite("RGB_Sharpen.png", image)
        else:
            print("no data[0x%x]" % ret)
        time.sleep(0.1)


if __name__ == "__main__":
    nDeviceNum = ctypes.c_uint(0)
    nDeviceNum_p = byref(nDeviceNum)
    # ch:获取设备数量 | en:Get device number
    ret = Mv3dRgbd.MV3D_RGBD_GetDeviceNumber(
        DeviceType_Ethernet
        | DeviceType_USB
        | DeviceType_Ethernet_Vir
        | DeviceType_USB_Vir,
        nDeviceNum_p,
    )
    if ret != 0:
        print("MV3D_RGBD_GetDeviceNumber fail! ret[0x%x]" % ret)
        os.system("pause")
        sys.exit()
    if nDeviceNum == 0:
        print("find no device!")
        os.system("pause")
        sys.exit()
    print("Find devices numbers:", nDeviceNum.value)

    stDeviceList = MV3D_RGBD_DEVICE_INFO_LIST()
    net = Mv3dRgbd.MV3D_RGBD_GetDeviceList(
        DeviceType_Ethernet
        | DeviceType_USB
        | DeviceType_Ethernet_Vir
        | DeviceType_USB_Vir,
        pointer(stDeviceList.DeviceInfo[0]),
        20,
        nDeviceNum_p,
    )

    for i in range(0, nDeviceNum.value):
        print("\ndevice: [%d]" % i)
        strModeName = ""
        for per in stDeviceList.DeviceInfo[i].chModelName:
            strModeName = strModeName + chr(per)
        print("device model name: %s" % strModeName)

        strSerialNumber = ""
        for per in stDeviceList.DeviceInfo[i].chSerialNumber:
            strSerialNumber = strSerialNumber + chr(per)
        print("device SerialNumber: %s" % strSerialNumber)

    # ch:创建相机示例 | en:Create a camera instance
    camera = Mv3dRgbd()
    # nConnectionNum = input("please input the number of the device to connect:")
    nConnectionNum = 0
    if int(nConnectionNum) >= nDeviceNum.value:
        print("intput error!")
        os.system("pause")
        sys.exit()

    # ch:打开设备 | en:Open device
    ret = camera.MV3D_RGBD_OpenDevice(
        pointer(stDeviceList.DeviceInfo[int(nConnectionNum)])
    )
    if ret != 0:
        print("MV3D_RGBD_OpenDevice fail! ret[0x%x]" % ret)
        os.system("pause")
        sys.exit()

    # ch:设置图像对齐深度图坐标系模式 | en:Set image align depth coordinate mode
    stParam = MV3D_RGBD_PARAM()
    stParam.enParamType = ParamType_Int
    stParam.ParamInfo.stIntParam.nCurValue = 1
    ret = camera.MV3D_RGBD_SetParam(MV3D_RGBD_INT_IMAGEALIGN, pointer(stParam))

    if MV3D_RGBD_OK != ret:
        print("SetParam fail! ret[0x%x]" % ret)
        camera.MV3D_RGBD_CloseDevice()
        os.system("pause")
        sys.exit()

    print("Set image align success.")

    # ch:获取传感器标定参数 | en:Get sensor calib param
    stCameraParam = MV3D_RGBD_CAMERA_PARAM()
    ret = camera.MV3D_RGBD_GetCameraParam(pointer(stCameraParam))
    if MV3D_RGBD_OK != ret:
        print("MV3D_RGBD_GetCameraParam fail! ret[0x%x]" % ret)
        camera.MV3D_RGBD_CloseDevice()
        os.system("pause")
        sys.exit()

    print("Get camera param success.")

    PrintCameraParam(stCameraParam)

    # ch:开始取流 | en:Start grabbing
    ret = camera.MV3D_RGBD_Start()
    if ret != 0:
        print("start fail! ret[0x%x]" % ret)
        camera.MV3D_RGBD_CloseDevice()
        os.system("pause")
        sys.exit()

    # ch:获取图像线程 | en:Get image thread
    try:
        hthreadhandle = threading.Thread(target=work_thread, args=(camera,))
        hthreadhandle.start()
    except:
        print("error: unable to start thread")

    hthreadhandle.join()
    # ch:停止取流 | en:Stop grabbing
    ret = camera.MV3D_RGBD_Stop()
    if ret != 0:
        print("stop fail! ret[0x%x]" % ret)
        os.system("pause")
        sys.exit()

    # ch:销毁句柄 | en:Destroy the device handle
    ret = camera.MV3D_RGBD_CloseDevice()
    if ret != 0:
        print("CloseDevice fail! ret[0x%x]" % ret)
        os.system("pause")
        sys.exit()

    sys.exit()
