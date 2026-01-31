import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

# 1. 初始化 RealSense 相机
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)
align_to = rs.stream.color
align = rs.align(align_to)

# 获取相机内参（用于 3D 坐标转换）
intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

# 2. 加载训练好的模型
model = YOLO(r'runs/pig_pose/yolov11_pig_kpt/weights/best.pt')


def get_3d_camera_coords(pixel_coords, depth_frame):
    """将像素坐标转换为 3D 空间坐标"""
    x, y = int(pixel_coords[0]), int(pixel_coords[1])
    dist = depth_frame.get_distance(x, y)
    if dist == 0: return None
    # 转换为 [X, Y, Z] (单位：米)
    point_3d = rs.rs2_deproject_pixel_to_point(intr, [x, y], dist)
    return point_3d


try:
    while True:
        # 获取对齐后的帧
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame: continue

        img = np.asanyarray(color_frame.get_data())
        overlay = img.copy()  # 用于绘制阴影层

        # 3. 模型推理
        results = model(img, stream=True, verbose=False)

        for r in results:
            if r.keypoints is None: continue

            # 获取关键点数据 (x, y, conf)
            # 索引参考: 7-withers, 0-5 为后肢点
            kpts = r.keypoints.data[0].cpu().numpy()

            # --- A. 处理肩胛部 (Withers, Index 7) ---
            withers_pt = kpts[7]
            if withers_pt[2] > 0.5:  # 置信度阈值
                # 绘制紫色阴影圆圈 (模拟分割)
                cv2.circle(overlay, (int(withers_pt[0]), int(withers_pt[1])), 40, (128, 0, 128), -1)
                # 获取 3D 坐标
                coord_3d = get_3d_camera_coords(withers_pt[:2], depth_frame)
                if coord_3d:
                    text = f"Withers: X:{coord_3d[0]:.2f} Y:{coord_3d[1]:.2f} Z:{coord_3d[2]:.2f}"
                    cv2.putText(img, text, (int(withers_pt[0]), int(withers_pt[1]) - 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # --- B. 处理后肢 (Back Legs, Index 0-5) ---
            leg_pts = kpts[0:6]
            valid_leg_pts = [p[:2] for p in leg_pts if p[2] > 0.5]
            if len(valid_leg_pts) >= 3:
                # 计算凸包多边形
                hull = cv2.convexHull(np.array(valid_leg_pts, dtype=np.int32))
                # 绘制黄色阴影覆盖 (模拟分割)
                cv2.fillPoly(overlay, [hull], (0, 255, 255))
                # 获取区域中心的 3D 坐标
                center_px = np.mean(valid_leg_pts, axis=0)
                coord_3d_leg = get_3d_camera_coords(center_px, depth_frame)
                if coord_3d_leg:
                    text_leg = f"BackLegs: Z:{coord_3d_leg[2]:.2f}m"
                    cv2.putText(img, text_leg, (int(center_px[0]), int(center_px[1])),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # 4. 融合阴影层与原图 (透明度 0.4)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        cv2.imshow('Pig Real-time Analysis', img)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
