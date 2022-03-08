import torch
from manopth.manolayer import ManoLayer
from manopth import demo
import numpy as np
import smoother

batch_size = 10
# Select number of principal components for pose space
ncomps = 23
_mano_root = 'mano/models'
# Initialize MANO layer
# mano_layer = ManoLayer(mano_root='mano/models', use_pca=True, ncomps=ncomps)
mano_layer = ManoLayer(flat_hand_mean=True,
                           side="left",
                           mano_root=_mano_root,
                           use_pca=False,
                           root_rot_mode='rotmat',
                           joint_rot_mode='rotmat'
                       )

view_mat = np.array([[1.0, 0.0, 0.0],
                     [0.0, 1.0, 0],
                     [0.0, 0, 1.0]])
one_mat =np.eye(3)
# Generate random shape parameters
# random_shape = torch.rand(batch_size, 10)
# Generate random pose parameters, including 3 values for global axis-angle rotation
# random_pose = torch.rand(batch_size, ncomps + 3)

random_pose = torch.load('temp_pose.pt')
random_shape = torch.load('temp_shape.pt')


# Forward pass through MANO layer
hand_verts, hand_joints = mano_layer(random_pose, random_shape.float())

hand_verts = hand_verts.clone().detach().cpu().numpy()[0]

# hand_verts = np.matmul(view_mat, hand_verts.T).T
# hand_verts = np.matmul(hand_verts,view_mat.T)
# hand_verts[:, 0] = hand_verts[:, 0] - 50
# hand_verts[:, 1] = hand_verts[:, 1] - 50
# new_tran = np.array([[-0.125 ,  0.6875,  0.    ]])
# mesh_tran = np.array([[-new_tran[0, 0], new_tran[0, 1], new_tran[0, 2]]])
# hand_verts = hand_verts - 100 * mesh_tran

new_verts_1 = torch.from_numpy(hand_verts)
new_verts_2 = torch.unsqueeze(new_verts_1,0)

#pose
hand_joints = hand_joints.clone().detach().cpu().numpy()[0]
tran_matrix = np.matmul(one_mat,view_mat)
# hand_joints = np.matmul(tran_matrix, hand_joints.T).T
new_joint1 = torch.from_numpy(hand_joints)
new_joint2 = torch.unsqueeze(new_joint1,0)

true_verts = torch.load('true_verts.pt')
new_true_verts = torch.unsqueeze(torch.from_numpy(true_verts),0)

demo.display_hand({'verts': new_verts_2, 'joints': new_joint2}, mano_faces=mano_layer.th_faces)

