import cv2
import torch
from manopth import manolayer
from model.detnet import detnet
from utils import func, bone, AIK, smoother
import numpy as np
import matplotlib.pyplot as plt
from utils import vis
from op_pso import PSO
import open3d

"""
导入mediapipe hand
"""
import mediapipe as mp
import time
mp_hands = mp.solutions.hands
# 导入模型
hands = mp_hands.Hands(static_image_mode=False,        # 是静态图片还是连续视频帧
                       max_num_hands=1,                # 最多检测几只手
                       min_detection_confidence=0.7,   # 置信度阈值
                       min_tracking_confidence=0.5)    # 追踪阈值
# 导入绘图函数
mpDraw = mp.solutions.drawing_utils
def process_frame(img):
    """
    使用blaze-hand 对人体的双手进行预测，返回相应手的坐标信息
    Args:
        img:

    Returns: img-图片；list_l :左手坐标；list_r:右手坐标

    """
    list_l, list_r = [],[]
    # 记录该帧开始处理的时间
    start_time = time.time()
    # 获取图像宽高
    h, w = img.shape[0], img.shape[1]

    # 将RGB图像输入模型，获取预测结果
    results = hands.process(img)

    if results.multi_hand_landmarks:  # 如果有检测到手

        handness_str = ''
        index_finger_tip_str = ''
        for hand_idx in range(len(results.multi_hand_landmarks)):
            temp_list = []
            # 获取该手的21个关键点坐标
            hand_21 = results.multi_hand_landmarks[hand_idx]

            # 可视化关键点及骨架连线
            mpDraw.draw_landmarks(img, hand_21, mp_hands.HAND_CONNECTIONS)

            # 记录左右手lable
            temp_handness = results.multi_handedness[hand_idx].classification[0].label
            handness_str += '{}:{} '.format(hand_idx, temp_handness)

            # 获取手腕根部深度坐标
            cz0 = hand_21.landmark[0].z

            for i in range(21):  # 遍历该手的21个关键点

                # 获取3D坐标

                cx = int(hand_21.landmark[i].x * w)
                cy = int(hand_21.landmark[i].y * h)
                cz = hand_21.landmark[i].z
                depth_z = cz0 - cz
                # 保存xyz
                temp_list.append(hand_21.landmark[i].x)
                temp_list.append(hand_21.landmark[i].y)
                temp_list.append(hand_21.landmark[i].z)

                # 用圆的半径反映深度大小
                radius = max(int(6 * (1 + depth_z * 5)), 0)

                if i == 0:  # 手腕
                    img = cv2.circle(img, (cx, cy), radius, (0, 0, 255), -1)
                if i == 8:  # 食指指尖
                    img = cv2.circle(img, (cx, cy), radius, (193, 182, 255), -1)
                    # 将相对于手腕的深度距离显示在画面中
                    index_finger_tip_str += '{}:{:.2f} '.format(hand_idx, depth_z)
                if i in [1, 5, 9, 13, 17]:  # 指根
                    img = cv2.circle(img, (cx, cy), radius, (16, 144, 247), -1)
                if i in [2, 6, 10, 14, 18]:  # 第一指节
                    img = cv2.circle(img, (cx, cy), radius, (1, 240, 255), -1)
                if i in [3, 7, 11, 15, 19]:  # 第二指节
                    img = cv2.circle(img, (cx, cy), radius, (140, 47, 240), -1)
                if i in [4, 12, 16, 20]:  # 指尖（除食指指尖）
                    img = cv2.circle(img, (cx, cy), radius, (223, 155, 60), -1)

                #保存左右手
            if temp_handness == "Left":
                list_l = temp_list
            else:
                list_r = temp_list


        scaler = 1
        img = cv2.putText(img, handness_str, (25 * scaler, 100 * scaler), cv2.FONT_HERSHEY_SIMPLEX, 1.25 * scaler,
                          (255, 0, 255), 2 * scaler)
        img = cv2.putText(img, index_finger_tip_str, (25 * scaler, 150 * scaler), cv2.FONT_HERSHEY_SIMPLEX,
                          1.25 * scaler, (255, 0, 255), 2 * scaler)

        # 记录该帧处理完毕的时间
        end_time = time.time()
        # 计算每秒处理图像帧数FPS
        FPS = 1 / (end_time - start_time + 0.0001)

        # 在图像上写FPS数值，参数依次为：图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细
        img = cv2.putText(img, 'FPS  ' + str(int(FPS)), (25 * scaler, 50 * scaler), cv2.FONT_HERSHEY_SIMPLEX,
                          1.25 * scaler, (255, 0, 255), 2 * scaler)


    return img, list_l, list_r

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
_mano_root = 'mano/models'

print('load model finished')
pose, shape = func.initiate("zero")
pre_useful_bone_len = np.zeros((1, 15))
pose0 = torch.eye(3).repeat(1, 16, 1, 1)

mano_l = manolayer.ManoLayer(flat_hand_mean=True,
                           side="left",
                           mano_root=_mano_root,
                           use_pca=False,
                           root_rot_mode='rotmat',
                           joint_rot_mode='rotmat')

print('start opencv')
point_fliter = smoother.OneEuroFilter(4.0, 0.0)
mesh_fliter = smoother.OneEuroFilter(4.0, 0.0)
shape_fliter = smoother.OneEuroFilter(4.0, 0.0)
cap = cv2.VideoCapture('./Video_data/left_hand_12345.avi')
# cap = cv2.VideoCapture(0)
print('opencv finished')
flag = 1
plt.ion()
f = plt.figure()

fliter_ax = f.add_subplot(111, projection='3d')
plt.show()
view_mat = np.array([[1.0, 0.0, 0.0],
                     [0.0, -1.0, 0],
                     [0.0, 0, -1.0]])

#left mesh
mesh = open3d.geometry.TriangleMesh()
hand_verts, j3d_recon = mano_l(pose0, shape.float())
mesh.triangles = open3d.utility.Vector3iVector(mano_l.th_faces)
hand_verts = hand_verts.clone().detach().cpu().numpy()[0]
mesh.vertices = open3d.utility.Vector3dVector(hand_verts)
viewer = open3d.visualization.Visualizer()
viewer.create_window(width=480, height=480, window_name='left')
viewer.add_geometry(mesh)
viewer.update_renderer()


print('start pose estimate')

pre_uv = None
shape_time = 0
opt_shape = None
shape_flag = True

#
array_left = np.array([])
array_right = np.array([])

while (cap.isOpened()):
    ret_flag, img = cap.read()
    if not ret_flag:
        break

    # 水平镜像翻转图像，使图中左右手与真实左右手对应
    # 参数 1：水平翻转，0：竖直翻转，-1：水平和竖直都翻转
    img = cv2.flip(img, 1)
    # BGR转RGB
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    cv2.imshow("Capture_Test", img)
    # result = module(input.unsqueeze(0))
    frame, list_l ,list_r = process_frame(img_RGB)
    if len(list_l) == 0 and len(list_r) == 0:
        continue

    # 开始替换左手
    if len(list_l) != 0:
        new_tran = np.array([[-0.125 ,  0.6875,  0.    ]])
        per_array = np.array(list_l)
        pre_joints = per_array.reshape(21,3)
        # pre_joints = pre_joints.clone().detach().cpu().numpy()

        flited_joints = point_fliter.process(pre_joints)

        fliter_ax.cla()

        filted_ax = vis.plot3d(flited_joints + new_tran, fliter_ax)
        pre_useful_bone_len = bone.caculate_length(pre_joints, label="useful")

        NGEN = 100
        popsize = 100
        low = np.zeros((1, 10)) - 3.0
        up = np.zeros((1, 10)) + 3.0
        parameters = [NGEN, popsize, low, up]
        pso = PSO(parameters, pre_useful_bone_len.reshape((1, 15)),_mano_root)
        pso.main()
        opt_shape = pso.ng_best
        opt_shape = shape_fliter.process(opt_shape)

        opt_tensor_shape = torch.tensor(opt_shape, dtype=torch.float)
        _, j3d_p0_ops = mano_l(pose0, opt_tensor_shape)
        template = j3d_p0_ops.cpu().numpy().squeeze(0) / 1000.0  # template, m 21*3
        ratio = np.linalg.norm(template[9] - template[0]) / np.linalg.norm(pre_joints[9] - pre_joints[0])
        j3d_pre_process = pre_joints * ratio  # template, m
        j3d_pre_process = j3d_pre_process - j3d_pre_process[0] + template[0]
        pose_R = AIK.adaptive_IK(template, j3d_pre_process)
        # save the left hand
        temp_array = np.squeeze(pose_R)
        if len(array_left)==0:
            array_left = temp_array
        else:
            array_left = np.append(array_left, temp_array, axis=0)

        pose_R = torch.from_numpy(pose_R).float()
        #  reconstruction  手指顺序显示是错误的，但实际是对的
        hand_verts, j3d_recon = mano_l(pose_R, opt_tensor_shape.float())
        mesh.triangles = open3d.utility.Vector3iVector(mano_l.th_faces)
        hand_verts = hand_verts.clone().detach().cpu().numpy()[0]
        hand_verts = mesh_fliter.process(hand_verts)
        hand_verts = np.matmul(view_mat, hand_verts.T).T
        hand_verts[:, 0] = hand_verts[:, 0] - 50
        hand_verts[:, 1] = hand_verts[:, 1] - 50
        mesh_tran = np.array([[-new_tran[0, 0], new_tran[0, 1], new_tran[0, 2]]])
        hand_verts = hand_verts - 100 * mesh_tran


        mesh.vertices = open3d.utility.Vector3dVector(hand_verts)
        mesh.paint_uniform_color([228 / 255, 178 / 255, 148 / 255])
        mesh.compute_triangle_normals()
        mesh.compute_vertex_normals()
        viewer.update_geometry(mesh)
        viewer.poll_events()


cap.release()
cv2.destroyAllWindows()

np.save('left_test_12345.npy',array_left)

