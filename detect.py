# -*- coding: utf-8 -*-
# @Time : 2024-11-2024/11/30 16:52
# @Author : 林枫
# @File : detect.py

import os
import cv2
from ultralytics import solutions


def workouts(model_path, video_path, point_list, up_angle=130.0, down_angle=90.0, show=True):
    cap = cv2.VideoCapture(video_path)

    assert cap.isOpened(), "Error reading video file"
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    out_path = "./runs/" + os.path.splitext(os.path.basename(video_path))[0] + ".avi"
    video_writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))  # 保存到指定位置

    gym = solutions.AIGym(
        model=model_path,
        show=show,
        line_width=2,
        up_angle=up_angle,
        down_angle=down_angle,
        kpts=point_list
    )

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print("Video frame is empty or video processing has been successfully completed.")
            break
        im0 = gym.monitor(im0)
        video_writer.write(im0)

    cv2.destroyAllWindows()
    video_writer.release()


if __name__ == '__main__':
    # model_path = "./weights/yolo11x-pose.pt"
    # video_path = "./video/俯卧撑.mp4"
    # point_list = [6, 8, 10]  # 头为正的俯卧撑，检测右肩、右肘、右手三个点形成的夹角。
    # workouts(model_path, video_path, point_list)

    # model_path = "./weights/yolo11x-pose.pt"
    # video_path = "./video/俯卧撑1.mp4"
    # point_list = [5, 7, 9]  # 头为右的俯卧撑，检测左肩、左肘、左手三个点形成的夹角。
    # workouts(model_path, video_path, point_list)

    # model_path = "./weights/yolo11x-pose.pt"
    # video_path = "./video/俯卧撑2.mp4"
    # point_list = [6, 8, 10]  # 头为右的俯卧撑，检测右肩、右肘、右手三个点形成的夹角。
    # workouts(model_path, video_path, point_list)

    # model_path = "./weights/yolo11x-pose.pt"
    # video_path = "./video/引体向上1.mp4"
    # point_list = [6, 8, 10]  # 头为正的引体向上，检测右肩、右肘、右手三个点形成的夹角。
    # workouts(model_path, video_path, point_list)

    # model_path = "./weights/yolo11x-pose.pt"
    # video_path = "./video/引体卷腹.mp4"
    # point_list = [6, 12, 14]  # 右侧朝向的引体卷腹，检测右肩、右腰、右膝三个点形成的夹角。
    # workouts(model_path, video_path, point_list)

    # model_path = "./weights/yolo11x-pose.pt"
    # video_path = "./video/引体卷腹1.mp4"
    # point_list = [5, 11, 13]  # 左侧朝向的引体卷腹，检测左肩、左腰、左膝三个点形成的夹角。
    # workouts(model_path, video_path, point_list)

    # model_path = "./weights/yolo11x-pose.pt"
    # video_path = "./video/引体卷腹2.mp4"
    # point_list = [6, 12, 14]  # 正面朝向的引体卷腹，检测右肩、右腰、右膝三个点形成的夹角。
    # workouts(model_path, video_path, point_list)

    # model_path = "./weights/yolo11x-pose.pt"
    # video_path = "./video/仰卧起坐.mp4"
    # point_list = [6, 12, 14]  # 左侧朝向的仰卧起坐，检测右肩、右腰、右膝三个点形成的夹角。
    # workouts(model_path, video_path, point_list)

    # model_path = "./weights/yolo11x-pose.pt"
    # video_path = "./video/V字卷腹.mp4"
    # point_list = [5, 11, 13]  # 右侧朝向的V字卷腹，检测左肩、左腰、左膝三个点形成的夹角。
    # workouts(model_path, video_path, point_list, up_angle=115)

    model_path = "./weights/yolo11x-pose.pt"
    video_path = "./video/深蹲.mp4"
    point_list = [11, 13, 15]  # 左侧朝向的深蹲，检测左肩、左腰、左膝三个点形成的夹角。
    workouts(model_path, video_path, point_list)
