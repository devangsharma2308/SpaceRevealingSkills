import time
import numpy as np
import math
from PIL import Image
import cv2
import os
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import open3d as o3d
from scipy.spatial.transform import Rotation as Rscipy
from sklearn.decomposition import PCA
import random

useNullSpace = 1
ikSolver = 0
pandaEndEffectorIndex = 11 #8
pandaNumDofs = 7

#ll = [-7]*pandaNumDofs

#Lower limits for null space (todo: set them to proper range)

ll = [
    -2.8973,  # Joint 1
    -1.7628,  # Joint 2
    -2.8973,  # Joint 3
    -3.0718,  # Joint 4
    -2.8973,  # Joint 5
    -0.0175,  # Joint 6
    -2.8973   # Joint 7
]


ul = [
    2.8973,   # Joint 1
    1.7628,   # Joint 2
    2.8973,   # Joint 3
    0.0698,   # Joint 4
    2.8973,   # Joint 5
    3.7525,   # Joint 6
    2.8973    # Joint 7
]

#ul = [7]*pandaNumDofs
#joint ranges for null space (todo: set them to proper range)
#jr = [7]*pandaNumDofs
#restposes for null space

jr = [
    5.7946,  # Joint 1
    3.5256,  # Joint 2
    5.7946,  # Joint 3
    3.1416,  # Joint 4
    5.7946,  # Joint 5
    3.77,    # Joint 6
    5.7946   # Joint 7
]

jointPositions=[0.98, 0.458, 0.31, -2.24, -0.30, 2.66, 2.32, 0.02, 0.02]
rp = jointPositions

class PandaSim(object):
  def __init__(self, bullet_client, offset, v_list,surface_flag):
    self.bullet_client = bullet_client
    self.offset = np.array(offset)
    self.bullet_client.configureDebugVisualizer(self.bullet_client.COV_ENABLE_GUI,1)

    #CAMERA DETAILS
    # cameraTargetPosition=[2.97,0.91,-1.31], cameraDistance=4.36, cameraPitch=-4
    self.bullet_client.resetDebugVisualizerCamera(cameraDistance=4.90, cameraYaw=93.60, cameraPitch=-4, cameraTargetPosition=[2.97,0.91,-1.31])

    self.view_matrix = self.bullet_client.computeViewMatrixFromYawPitchRoll(
    cameraTargetPosition=[2.97,0.91,-1.31],
    distance=4.90,
    yaw=93.60,
    pitch=-4,
    roll=0,            
    upAxisIndex=1      # Y is up in PyBullet
    )

    self.projection_matrix = self.bullet_client.computeProjectionMatrixFOV(
    fov=60.0,         # Standard depth camera FOV
    aspect=320/200,       
    nearVal=0.01,     # Near clipping plane
    farVal=10.0       # Far clipping plane
    )



    #print("offset=",offset)
    flags = self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
    legos=[]
    
    # self.bullet_client.loadURDF("tray/traybox.urdf", [0+offset[0], 0+offset[1], -0.6+offset[2]], [-0.5, -0.5, -0.5, 0.5], flags=flags)
    # legos.append(self.bullet_client.loadURDF("lego/lego.urdf",np.array([0.1, 0.3, -0.5])+self.offset, flags=flags))
    # legos.append(self.bullet_client.loadURDF("lego/lego.urdf",np.array([-0.1, 0.3, -0.5])+self.offset, flags=flags))
    # legos.append(self.bullet_client.loadURDF("lego/lego.urdf",np.array([0.1, 0.3, -0.7])+self.offset, flags=flags))
    # sphereId = self.bullet_client.loadURDF("sphere_small.urdf",np.array( [0, 0.3, -0.6])+self.offset, flags=flags)
    # self.bullet_client.loadURDF("sphere_small.urdf",np.array( [0, 0.3, -0.5])+self.offset, flags=flags)
    # self.bullet_client.loadURDF("sphere_small.urdf",np.array( [0, 0.3, -0.7])+self.offset, flags=flags)

    #orn_eul_mani = [-math.pi/2,math.pi/2,0]
    orn=[-0.707107, 0.0, 0.0, 0.707107]#p.getQuaternionFromEuler([-math.pi/2,math.pi/2,0])


    eul = self.bullet_client.getEulerFromQuaternion([-0.5, -0.5, -0.5, 0.5])
    

    
    self.panda_pos = np.array([-1.0,0.625,-0.725])
    self.panda = self.bullet_client.loadURDF("franka_panda/panda.urdf", self.panda_pos+self.offset, orn, useFixedBase=True, flags=flags)
    # urdf_path = "d:/Project/small_setup/small_setup/data/ur5/ur5.urdf"
    # self.panda = self.bullet_client.loadURDF(urdf_path, self.panda_pos+self.offset, orn, useFixedBase=True, flags=flags)
    
    self.bullet_client.loadURDF("iitd_table/table.urdf", [-0.625+offset[0], 0.625+offset[1], -1.7+offset[2]], [-0.707107, 0.0, 0.0, 0.707107], flags=flags)
    self.door = self.bullet_client.loadURDF("iitd_wood_door/wood_door.urdf", [0.625+offset[0], 0.625+offset[1], -1.5+offset[2]], [-0.707107, -0.707107, -0.707107,  0.707107], flags=flags)
    self.plane=self.bullet_client.loadURDF("iitd_plane/plane.urdf", [1.0+offset[0], 3.5+offset[1], -1.0+offset[2]], [-0.707107, -0.707107, -0.707107, 0.707107], flags=flags)
    # self.bullet_client.loadURDF("pass_way/pass_way.urdf", [0.625+offset[0], 0.625+offset[1], -1.8+offset[2]], [-0.707107, -0.707107, -0.707107,  0.707107], flags=flags)
    # self.box=self.bullet_client.loadURDF("iitd_50cm_box/50cm_box.urdf", [-0.4+offset[0], 0.625+offset[1], -0.6+offset[2]], [-0.707107, 0.0, 0.0, 0.707107], flags=flags)

    
    # time.sleep(100)
    self.box_dim = [0.5,0.5,0.8]
    # self.box_dim = [0.5,0.125,0.8] #THIN
    # self.box_dim = [0.5,0.1875,0.8] #THIN
    # self.box_dim = [0.5,1,0.4] #THICK
    # self.box_dim = [0.25,0.25,0.4]
    self.box_pos = [-0.35+offset[0], 0.625+offset[1], -0.7+offset[2]]  #y=0.625 is table top and 0.4 is half box dim in Z
    # self.box_pos = [-0.35+offset[0], 0.625+offset[1], -0.4+offset[2]] # for thick box
    # self.box_pos = [-0.4+offset[0], 0.625+offset[1], -0.7+offset[2]]
    self.box_orn_euler = [-math.pi/2.,0*math.pi/4,0.]


    box_orn = self.bullet_client.getQuaternionFromEuler(self.box_orn_euler) #Rotate about y-axis keep others fixed
    eul = self.bullet_client.getEulerFromQuaternion([-0.707107, 0, 0.0, 0.707107])
    #print(eul)
    #self.box=self.bullet_client.loadURDF("iitd_80cm_box/80cm_box.urdf", [-0.4+offset[0], 0.625+offset[1], -1.0+offset[2]], [-0.707107, 0, 0.0, 0.707107], flags=flags)
    #LOADING THE BOX
    self.box=self.bullet_client.loadURDF("iitd_80cm_box/80cm_box.urdf", self.box_pos, box_orn, flags=flags)
    # time.sleep(100)
    # self.box=self.bullet_client.loadURDF("iitd_80cm_box/80cm_box_2.urdf", self.box_pos, box_orn, flags=flags)
    # self.box=self.bullet_client.loadURDF("iitd_80cm_box/80cm_box_3.urdf", self.box_pos, box_orn, flags=flags)
    # self.box=self.bullet_client.loadURDF("iitd_80cm_box/80cm_box_4.urdf", self.box_pos, box_orn, flags=flags)
    # self.box=self.bullet_client.loadURDF("iitd_80cm_box/80cm_box_5.urdf", self.box_pos, box_orn, flags=flags)

    # box_pos = [-0.35+offset[0], 0.625+offset[1], -0.6+offset[2]] 
    # box_orn_euler = [-0*math.pi/2.,0*math.pi/4,0.]
    # box_orn = self.bullet_client.getQuaternionFromEuler(box_orn_euler) #Rotate about y-axis keep others fixed
    # self.box=self.bullet_client.loadURDF("iitd_80cm_box/80cm_box_6.urdf", self.box_pos, box_orn, flags=flags)
    # time.sleep(1000)

    ##########################INITIALIZATION FOR THOMPSON SAMPLING##################################

    
    self.thomp_start = False
    self.action_taken = False
    self.play_phase_length = 5
    self.v_list = v_list
    self.current_v = self.v_list[0]

    self.v_rewards = {v: 0 for v in self.v_list}
    self.v_rewards_sum = {v: 0 for v in self.v_list}
    self.v_rewards_count = {v: 0 for v in self.v_list}

    self.current_v_phase_start = None
    self.contact_force  = []

    self.play_phase_start_idx = math.inf
    self.play_phase_over_idx = math.inf
  

    self.box_centroid_list = []
    self.box_orientation_list = []
    self.box_origin_list = [] # LOCAL FRAME OF THE BOX
    self.point_on_tilting_axis_list = []
    self.rewards_list_play_idx_change = []


    # self.action_space = list(itertools.product(self.action_u, self.forces))
    # self.chosen_action = self.action_space[0]
    
    # self.action_to_alpha_beta = {action: [1, 1] for action in self.action_space}

    self.thomp_actions_selected = []
    #######################################################################################################

    
    b_p,b_o = self.get_box_position()
    # print('*****************')
    print(self.box_pos,self.box_orn_euler)

    # print('--------------------')
    b_o = self.bullet_client.getEulerFromQuaternion(b_o)
    # print(b_p,b_o)
    # print('--------------------')

    self.dist_from_origin = np.array(b_p)-np.array(self.box_pos)
    # print(self.dist_from_origin)

    # print('$$$$$$$$$$$$$$$$$$$$$$$')
    box_p,box_o = self.get_box_origin()
    # print(box_p,box_o)

    box_corners_world = self.get_box_corner_points(self.box_dim,self.box_pos,self.box_orn_euler)#HERE IS THE PROBLEM
    #print(box_corners_world)

    surface_1 = [box_corners_world[2],box_corners_world[3],box_corners_world[6],box_corners_world[7]]
    surface_2 = [box_corners_world[3],box_corners_world[0],box_corners_world[7],box_corners_world[4]]
    surface_3 = [box_corners_world[0],box_corners_world[1],box_corners_world[4],box_corners_world[5]]
    surface_4 = [box_corners_world[1],box_corners_world[2],box_corners_world[5],box_corners_world[6]]

    # time.sleep(1000)
    surface = surface_3
    self.sur_flag = 3

    if surface_flag==1:
      surface = surface_1
      self.sur_flag = 1
    elif surface_flag==2:
      surface = surface_2
      self.sur_flag = 2
    elif surface_flag==3:
      surface = surface_3
      self.sur_flag = 3
    elif surface_flag==4:
      surface = surface_3
      self.sur_flag = 3
    else:
      print('INCORRECT SURFACE')


    self.normal = self.get_unit_normal(surface)
    # print('Normal Direction = ',self.normal)

    # point_on_sur = self.get_point_on_surface(surface,0.5,1)+np.array([0,0.4,0])
    point_on_sur = self.get_point_on_surface(surface,0.5,self.current_v)
    self.st_point  = point_on_sur-0.1*self.normal

    # print('Point on surface = ',point_on_sur)
    if self.sur_flag == 1:
      # target_orientation_euler = [0*math.pi/2,0*math.pi/4, 0]
      target_orientation_euler = [0*math.pi/2,0*math.pi/2, 0*math.pi/2]
    elif self.sur_flag == 2:
      target_orientation_euler = [0*math.pi/2,math.pi/2, 0]
    elif self.sur_flag == 3:
      target_orientation_euler = [0*math.pi/2,math.pi, 0]
    
    self.target_orientation = self.bullet_client.getQuaternionFromEuler(target_orientation_euler)

    end_effector_state = self.bullet_client.getLinkState(self.panda, pandaEndEffectorIndex)
    current_position = end_effector_state[0]
    # print(current_position)

    #intermediate_positions for surface 1
    # intermediate_positions = [[self.panda_pos[0]+0.3,0.975,self.panda_pos[0]],[self.panda_pos[0]+0.3,0.975,self.panda_pos[0]-0.175],[self.st_point[0],self.st_point[1],self.panda_pos[0]-0.175],[box_pos[0],0.975,self.panda_pos[0]-0.175]]
    # intermediate_positions = [[current_position[0]+0.2,current_position[1]-0.2,current_position[2]],[intermediate_positions[0][0],intermediate_positions[0][1],intermediate_positions[0][2]-0.2],[self.st_point[0],intermediate_positions[1][1],intermediate_positions[1][2]],[intermediate_positions[2][0],self.st_point[1],intermediate_positions[2][2]]]

    if self.sur_flag==1:
      # Step 1: 
      intermediate_positions = [[current_position[0]+0.2, current_position[1]-0.2, current_position[2]]]
      # Step 2:
      intermediate_positions.append([intermediate_positions[0][0], intermediate_positions[0][1], intermediate_positions[0][2] - 0.4])
      # Step 3: 
      intermediate_positions.append([self.st_point[0], intermediate_positions[1][1], intermediate_positions[1][2]])
      # Step 4: 
      intermediate_positions.append([intermediate_positions[2][0], self.st_point[1], intermediate_positions[2][2]])

    elif self.sur_flag == 3:
       
       # Step 1: 
      intermediate_positions = [[current_position[0]+0.2, current_position[1]-0.2, current_position[2]]]
      # Step 2:
      intermediate_positions.append([intermediate_positions[0][0], intermediate_positions[0][1], intermediate_positions[0][2] + 0.6])
      # Step 3: 
      intermediate_positions.append([self.st_point[0], intermediate_positions[1][1], intermediate_positions[1][2]])
      # Step 4: 
      intermediate_positions.append([intermediate_positions[2][0], self.st_point[1], intermediate_positions[2][2]])
       
       
       
  
    #intermediate_positions for surface 2
    #intermediate_positions = [[self.panda_pos[0]+0.3,0.975,self.panda_pos[0]],[self.panda_pos[0]+0.3,0.975,self.st_point[2]],[self.panda_pos[0]+0.3,self.st_point[1],self.st_point[2]]]


    self.set_end_effector_positions_with_final_orientation(intermediate_positions,self.st_point,self.target_orientation)
    # self.set_end_effector_initial_position(self.st_point,self.target_orientation)
    # self.check_end_effector_position(self.st_point)

    # print('------------@@@@@@@@@@@@------------------------')
    # print("Target Position:", self.st_point)
    # end_effector_state = self.bullet_client.getLinkState(self.panda, pandaEndEffectorIndex)
    # print("Current Position:", end_effector_state[0])
    # print('------------@@@@@@@@@@@@------------------------')

    # time.sleep(1000)

    #Area for Reward
    self.right_edge_pt = [self.panda_pos[0],0,self.panda_pos[2]+0.8]
    self.left_edge_pt = [self.panda_pos[0],0,self.panda_pos[2]-0.8]
    self.center_line_pt = [self.panda_pos[0],0,self.panda_pos[2]]

    angle = 0 #Degrees from x axis towards z axis
    
    self.right_edge_dir = np.array([math.cos(math.radians(angle)),0,math.sin(math.radians(angle))])
    self.left_edge_dir = np.array([math.cos(math.radians(angle)),0,-math.sin(math.radians(angle))])
    self.center_line_dir = np.array([1,0,0])

    # print('edges directions = ',self.left_edge_dir,self.right_edge_dir)


    # box_position, box_orientation = self.get_box_position()
    # box_position_np = np.array(box_position)
    # print('BOX_POSiTION = ',box_position_np)
    # print('ORIGINAL = ',box_pos)

    # projeceted_point_lt = self.project_point_on_line(box_position,self.left_edge_pt,self.left_edge_dir)
    # projeceted_point_rt = self.project_point_on_line(box_position,self.right_edge_pt,self.right_edge_dir)

    # inter_left = self.intersection_of_lines(box_position,projeceted_point_lt,self.center_line_pt,self.center_line_dir)
    # inter_right = self.intersection_of_lines(box_position,projeceted_point_rt,self.center_line_pt,self.center_line_dir)

    # print(inter_left)
    # print(inter_right)


    self.t = 0.
    self.i=0
    self.frames = []
    self.rewards_list =[]
    self.visual_reward_list = []
    self.window = None
    self.policy = 1
    self.step_size = 0.08

    self.total_joints =  self.bullet_client.getNumJoints(self.panda)
    self.linkIndex = [i for i in range(-1, self.total_joints)]
    # print(self.linkIndex)
    self.images = []
    self.plots = []
    self.rgb_image_folder = "cv_images"
    self.plt_image_folder = "plt_images"


    self.action_u = [0.1,0.3,0.5,0.7,0.9]
    # self.action_u = [0.5]
    # self.forces = [50,80,100,200]
    self.forces = [200]
    self.force_index = 0
    self.force_counter = 1
    self.forces_list=[]

    self.sequence_index = 0
    self.sequence_complete = False

    ################################################################################################
    self.thetas = [0]

    self.orn_bins = [
      (-math.pi/4, -math.pi/6),
      (-math.pi/6, -math.pi/12),
      (-math.pi/12, math.pi/12),
      (math.pi/12, math.pi/6),
      (math.pi/6, math.pi/4)
      ]
    
    self.orn_bin_index = self.find_orn_bin_index(self.box_orn_euler[1],self.orn_bins)

    # print(self.orn_bin_index)

    if not os.path.exists(self.rgb_image_folder):
      os.makedirs(self.rgb_image_folder)

    if not os.path.exists(self.plt_image_folder):
      os.makedirs(self.plt_image_folder)


  def reset(self):
    pass
  

  def reward_function(self):

    box_position, box_orientation = self.get_box_position()

    living_reward = -10
    

    box_orientation=self.bullet_client.getEulerFromQuaternion(box_orientation)
    # print('box_ori = ',box_orientation)

    # box_corners_world = self.get_box_corner_points([.5,.5,.8],box_position,box_orientation)#HERE IS THE PROBLEM
    # #print(box_corners_world)
    # surface_1 = [box_corners_world[2],box_corners_world[3],box_corners_world[6],box_corners_world[7]]
    # print(surface_1)

    # time.sleep(1000)


    box_position_np = np.array(box_position)
    box_position_np[1] =0 #Projecting in XZ plane

    flag=0

    diff = self.center_line_pt[2]-box_position_np[2]

    if diff<=0:
      #Right Side
      projeceted_point_rt = self.project_point_on_line(box_position_np,self.right_edge_pt,self.right_edge_dir)
      inter_right = self.intersection_of_lines(box_position_np,projeceted_point_rt,self.center_line_pt,self.center_line_dir)

      l = np.linalg.norm(box_position_np - projeceted_point_rt)
      L = np.linalg.norm(inter_right - projeceted_point_rt)

      if projeceted_point_rt[2]<=box_position_np[2]:
         flag=1

    else:
      #Left Side
      projeceted_point_lt = self.project_point_on_line(box_position_np,self.left_edge_pt,self.left_edge_dir)
      inter_left = self.intersection_of_lines(box_position_np,projeceted_point_lt,self.center_line_pt,self.center_line_dir)

      l = np.linalg.norm(box_position_np - projeceted_point_lt)
      L = np.linalg.norm(inter_left - projeceted_point_lt)

      if projeceted_point_lt[2]>=box_position_np[2]:
         flag=1

    if flag==0:
       reward = 100*(1-l/(L+1e-10))+living_reward
    else:
       reward = 100

    return reward


  def intersection_of_lines(self,P1, P2, Q1, L2_dir):#P1 is the box cm,P2 is projection of box on the edge, Q1 is centerline st point,L2_dir is its dirn.
   
    #Line 1 direction vector
    L1_dir = P2 - P1
    
    #Set up the system of equations
    A = np.array([L1_dir, -L2_dir]).T  #A 3x2 matrix
    b = Q1 - P1                        #A 3x1 vector
    
    #Solve for t and s
    t, s = np.linalg.solve(A.T @ A, A.T @ b)  
    
    #Intersection point on line 1 (since they intersect)
    intersection_point = P1 + t * L1_dir
    
    return intersection_point


  def project_point_on_line(self,P, L_start, L_dir):
    #Vector from L_start to P
    v = P - L_start
    
    #Projection of vector v onto the direction vector L_dir
    projection_length = np.dot(v, L_dir) / np.dot(L_dir, L_dir)
    projection_vector = projection_length * L_dir
    
    #The projected point on the line
    projected_point = L_start + projection_vector

    return projected_point


  def get_box_position(self):
    #Get the current position and orientation of the box
    box_position, box_orientation = self.bullet_client.getBasePositionAndOrientation(self.box)
    return box_position, box_orientation

  
  def set_end_effector_initial_position(self, target_position,target_orientation):
    # Use inverse kinematics to get the joint angles for the desired end-effector position
    joint_angles = self.bullet_client.calculateInverseKinematics(
        self.panda, 
        pandaEndEffectorIndex,  # End-effector index
        target_position,        # Target position
        targetOrientation=target_orientation,
        maxNumIterations=10000, residualThreshold=0.0001
    )

    # Set the joint angles to move the robot to the calculated position
    for i in range(pandaNumDofs):
      self.bullet_client.resetJointState(self.panda, i, joint_angles[i])

    # index = 0
    # for j in range(self.bullet_client.getNumJoints(self.panda)):
    #   self.bullet_client.changeDynamics(self.panda, j, linearDamping=0, angularDamping=0)
    #   info = self.bullet_client.getJointInfo(self.panda, j)
  
    #   jointName = info[1]
    #   jointType = info[2]
    #   if (jointType == self.bullet_client.JOINT_PRISMATIC):
        
    #     self.bullet_client.resetJointState(self.panda, j, joint_angles[index]) 
    #     index=index+1
    #   if (jointType == self.bullet_client.JOINT_REVOLUTE):
    #     self.bullet_client.resetJointState(self.panda, j, joint_angles[index]) 
    #     index=index+1

  def set_end_effector_positions_with_final_orientation(self, intermediate_positions, target_position, target_orientation):
    
    current_orientation = self.bullet_client.getLinkState(self.panda, pandaEndEffectorIndex)[5]
    
    #Move through all intermediate positions, keeping the current orientation
    for position in intermediate_positions:
        
        # joint_angles = self.bullet_client.calculateInverseKinematics(
        #     self.panda, 
        #     pandaEndEffectorIndex,  # End-effector index
        #     position,               # Current intermediate position
        #     targetOrientation=current_orientation,  # Keep the current orientation
        #     maxNumIterations=1000, residualThreshold=0.0001
        # )

        end_effector_state = self.bullet_client.getLinkState(self.panda, pandaEndEffectorIndex)
        current_position = end_effector_state[0]

        for new_point in self.generate_points(current_position,position,10):
        

          joint_angles = self.bullet_client.calculateInverseKinematics(
              self.panda, 
              pandaEndEffectorIndex,  # End-effector index
              new_point,               # Current intermediate position
              targetOrientation=current_orientation,  # Keep the current orientation
              maxNumIterations=1000, residualThreshold=0.0001
          )
          for i in range(pandaNumDofs):
            self.bullet_client.resetJointState(self.panda, i, joint_angles[i])
            

          time.sleep(0.02)



        # Set the joint angles to move the robot to the calculated position
        # for i in range(pandaNumDofs):
        #     self.bullet_client.resetJointState(self.panda, i, joint_angles[i])
        
        
        # time.sleep(3)  
    end_effector_state = self.bullet_client.getLinkState(self.panda, pandaEndEffectorIndex)
    current_position = end_effector_state[0]

    for new_point in self.generate_points(current_position,target_position,10):
    # Finally, move to the target position and apply the target orientation
      joint_angles = self.bullet_client.calculateInverseKinematics(
          self.panda, 
          pandaEndEffectorIndex,  # End-effector index
          new_point,        # Final target position
          targetOrientation=target_orientation,  # Apply the target orientation only at the end
          maxNumIterations=1000, residualThreshold=0.0001
      )
      time.sleep(0.05)
    
    for i in range(pandaNumDofs):
        self.bullet_client.resetJointState(self.panda, i, joint_angles[i])

    # self.bullet_client.setJointMotorControlArray(
    # bodyIndex=self.panda, 
    # jointIndices=list(range(pandaNumDofs)), 
    # controlMode=self.bullet_client.POSITION_CONTROL, 
    # targetPositions=joint_angles[:7]
    # )


  def check_end_effector_position(self, target_position):
    # Get the current end-effector position using getLinkState
    end_effector_state = self.bullet_client.getLinkState(self.panda, pandaEndEffectorIndex)
    current_position = end_effector_state[0]  # Position is the first element of the returned tuple

    print('--------------------------------------------')
    print(f"Target Position: {target_position}")
    print(f"Current Position: {current_position}")
    print('--------------------------------------------')


    # Check if the current position is close enough to the target position
    if np.allclose(current_position, target_position, atol=1e-3):
        print("End-effector is at the correct position.")
    else:
        print("End-effector is NOT at the correct position.")

  # def get_box_corner_points(self,box_dims, box_position, euler_angles):
   
  #   length_x, length_y, length_z = box_dims
    
  #   local_corners = np.array([
  #       [-length_x/2, -length_y/2, -length_z/2],
  #       [length_x/2, -length_y/2, -length_z/2],
  #       [length_x/2, length_y/2, -length_z/2],
  #       [-length_x/2, length_y/2, -length_z/2],
  #       [-length_x/2, -length_y/2, length_z/2],
  #       [length_x/2, -length_y/2, length_z/2],
  #       [length_x/2, length_y/2, length_z/2],
  #       [-length_x/2, length_y/2, length_z/2]
  #   ])
    
  #   box_orn = self.bullet_client.getQuaternionFromEuler(euler_angles)

  #   world_corners = []
    
  #   for corner in local_corners:
      
  #     rotated_corner = self.bullet_client.multiplyTransforms([0, 0, 0], box_orn, corner, [0, 0, 0, 1])[0]
        
  #     world_corner = np.array(rotated_corner) + np.array(box_position)
  #     world_corners.append(world_corner)

  #   return world_corners


  # def get_box_corner_points(self, box_dims, box_position, euler_angles):
  #   length_x, length_y, length_z = box_dims
    
    
  #   local_corners = np.array([
  #       [-length_x / 2, -length_y / 2, -length_z / 2],
  #       [length_x / 2, -length_y / 2, -length_z / 2],
  #       [length_x / 2, length_y / 2, -length_z / 2],
  #       [-length_x / 2, length_y / 2, -length_z / 2],
  #       [-length_x / 2, -length_y / 2, length_z / 2],
  #       [length_x / 2, -length_y / 2, length_z / 2],
  #       [length_x / 2, length_y / 2, length_z / 2],
  #       [-length_x / 2, length_y / 2, length_z / 2]
  #   ])
    
    
  #   box_orn = self.bullet_client.getQuaternionFromEuler(euler_angles)
    
    
  #   world_corners = []
    
    
  #   for corner in local_corners:
        
  #       rotated_corner, _ = self.bullet_client.multiplyTransforms([0, 0, 0], box_orn, corner, [0, 0, 0, 1])
        
        
  #       world_corner = np.array(rotated_corner) + np.array(box_position)
  #       world_corners.append(world_corner)

  #   return np.array(world_corners)

  def get_box_corner_points(self, box_dims, box_position, euler_angles):
    length_x, length_y, length_z = box_dims
    
    # Define local corners with bottom center as origin
    local_corners = np.array([
        [-length_x / 2, -length_y / 2, 0],
        [length_x / 2, -length_y / 2, 0],
        [length_x / 2, length_y / 2, 0],
        [-length_x / 2, length_y / 2, 0],
        [-length_x / 2, -length_y / 2, length_z],
        [length_x / 2, -length_y / 2, length_z],
        [length_x / 2, length_y / 2, length_z],
        [-length_x / 2, length_y / 2, length_z]
    ])
    
    # Get quaternion for the box orientation
    box_orn = self.bullet_client.getQuaternionFromEuler(euler_angles)
    
    world_corners = []
    
    # Rotate and translate each corner
    for corner in local_corners:
        rotated_corner, _ = self.bullet_client.multiplyTransforms([0, 0, 0], box_orn, corner, [0, 0, 0, 1])
        world_corner = np.array(rotated_corner) + np.array(box_position)
        world_corners.append(world_corner)

    return np.array(world_corners)


  def get_tipping_point_center(self, box_dims, box_position, euler_angles):
    """
    Returns the tipping point at the center of the tipping edge based on surflag.
    
    """
    # Get all 8 world corners
    world_corners = self.get_box_corner_points(box_dims, box_position, euler_angles)
    # print(world_corners)

    if self.sur_flag == 1:
        # Right edge: between corners 0 and 1
        p1 = world_corners[0]
        p2 = world_corners[1]
    elif self.sur_flag == 3:
        # Left edge: between corners 2 and 3
        p1 = world_corners[2]
        p2 = world_corners[3]
    

    tipping_point = (p1 + p2) / 2
    return tipping_point

  def get_unit_normal(self,surface):
    p1 = surface[0]
    p2 = surface[1]
    p3 = surface[2]
    p4 = surface[3]

    v1 = p2-p1
    v2 = p4-p1

    normal = np.cross(v2,v1)

    magnitude = np.linalg.norm(normal)
    unit_normal = normal / magnitude

    return unit_normal


  def get_point_on_surface(self,surface, u, v):
   
    p1, p2, p3, p4 = surface[0], surface[1], surface[3], surface[2]
    # print(p1,p2,p3,p4)
    
    # Bilinear interpolation formula
    point = (1 - u) * (1 - v) * p1 + u * (1 - v) * p2 + u * v * p3 + (1 - u) * v * p4
    
    return point


  def generate_points(self,start_point, end_point, num_points):
    start_point = np.array(start_point)
    end_point = np.array(end_point)
    step = (end_point - start_point) / (num_points - 1)
    current_point = start_point
    
    yield current_point
    
    for _ in range(1, num_points):
        current_point = current_point + step
        yield current_point
    
    # # Step the simulation to allow the robot to reach the position
    # for _ in range(240):  # Adjust this loop to ensure the robot reaches the position
    #     self.bullet_client.stepSimulation()
    #     time.sleep(1./240.)

  def print_end_effector_pose(self):
    # Get the current joint angles (positions)
    joint_positions = []

    for i in range(pandaNumDofs):
      joint_state = self.bullet_client.getJointState(self.panda, i)
      joint_positions.append(joint_state[0])

    # Calculate the position and orientation of the end effector
    end_effector_state = self.bullet_client.getLinkState(self.panda, pandaEndEffectorIndex)
    end_effector_position = end_effector_state[4]
    end_effector_orientation = end_effector_state[5]
    end_effector_orientation_euler = self.bullet_client.getEulerFromQuaternion(end_effector_orientation)

    print("End Effector Position: [x, y, z] =", end_effector_position)
    print("End Effector Orientation (Quaternion): [x, y, z, w] =", end_effector_orientation)
    print("End Effector Orientation (Euler): [roll, pitch, yaw] =", end_effector_orientation_euler)


  def check_collision(self):
    # Get all contact points involving the robot
    contact_points = self.bullet_client.getContactPoints(bodyA=self.panda)

    # Check if any contact points are detected
    if len(contact_points) > 0:
        print("Collision detected!")
        for contact in contact_points:
            print(f"Contact with object ID: {contact[2]} at position: {contact[5]}")
        return True
    else:
        print("No collision detected.")
        return False

  def rotate_point_y_axis(self,point, angle):
    #angle in radians
    rotation_matrix = np.array([[np.cos(angle), 0, np.sin(angle)], 
                                [0, 1, 0],  
                                [-np.sin(angle), 0, np.cos(angle)]])
    return np.dot(rotation_matrix, point)

  def square_vertices_xz_plane(self,centroid, side_length, angle):
      half_side = side_length / 2
      
      vertices = np.array([
          [-half_side, 0, -half_side],  
          [half_side, 0, -half_side],   
          [half_side, 0, half_side],    
          [-half_side, 0, half_side]    
      ])
      
      rotated_vertices = np.array([self.rotate_point_y_axis(v, angle) for v in vertices])
      rotated_vertices += np.array(centroid) 
      
      return rotated_vertices

  def unit_normals_to_centroid_xz(self,vertices, centroid):
      num_vertices = len(vertices)
      normals = []
      
      for i in range(num_vertices):
          
          next_i = (i + 1) % num_vertices
          midpoint = (vertices[i] + vertices[next_i]) / 2
          
          vector_to_centroid = centroid - midpoint
          
          unit_normal = vector_to_centroid / np.linalg.norm(vector_to_centroid)
          normals.append(unit_normal)
      
      return np.array(normals)

  def save_plots(self,plots_list):
    i=1
    for img in plots_list:
      plt.imshow(img)
      plot_filename = os.path.join(self.plt_image_folder, f"plot{self.i}.png")  
      plt.savefig(plot_filename)
      i+=1
  
  def get_window(self,gray_scale_image):
    new = np.zeros_like(gray_scale_image)

    for i in range(new.shape[0]):
      for j in range(new.shape[1]):
          if gray_scale_image[i,j]<=5:
            new[i,j] = 255


    df = pd.DataFrame(new)
    df.to_csv('out_new.csv', index=False, header=False) 

    #Sum along the columns
    sum_columns = np.sum(new, axis=0)
    max_column_value = np.max(sum_columns)


    counter = 0

    for i in range(sum_columns.shape[0]):
      if sum_columns[i]>=max_column_value/4 and counter==0:
        counter=1
      if counter==1 and sum_columns[i]<=max_column_value/4:
         window_y_start = i
         counter=2
      if counter==2 and sum_columns[i]>=max_column_value/4:
         window_y_end = i
         counter=3
        
    # print(window_y_start,window_y_end)
        
    # plt.plot(sum_columns); plt.title('Sum of rows along the column'); plt.show()
    # plt.close()
    # column_gradient_vector = np.gradient(sum_columns)
    # print(column_gradient_vector)
    #Sum along the rows
    sum_rows = np.sum(new, axis=1)
    max_row_value = np.max(sum_rows)

    for i in range(sum_rows.shape[0]):
      if sum_rows[i]>=max_row_value/4:
        window_x_start = i
        break

    for i in range(sum_rows.shape[0]-1,0,-1):
      if sum_rows[i]>=max_row_value/4:
        window_x_end = i
        break

    # print(window_x_start,window_x_end)
      

    # plt.plot(sum_rows); plt.title('Sum of columns along the row'); plt.show()
    # plt.close()
    # row_gradient_vector = np.gradient(sum_rows)
    # print(row_gradient_vector)
    window  = [window_x_start, window_x_end,window_y_start,window_y_end]   

    return window

  def rotate_vector_90_deg_y(self,vector):
    rotation_matrix = np.array([
        [0, 0, -1],
        [0, 1, 0],
        [1, 0, 0]
    ])
    
    rotated_vector = rotation_matrix @ vector
    return rotated_vector
  
  def get_box_origin(self):
     
    box_position, box_orientation = self.get_box_position()
    box_orientation=self.bullet_client.getEulerFromQuaternion(box_orientation)

    box_position = np.array(box_position) - self.dist_from_origin

    return box_position,box_orientation
  

  def move_to_point_on_box(self,current_position,orn,normal,step_size,u,v):
    box_origin_pos, box_origin_orn_eul = self.get_box_origin()

    box_corners_world = self.get_box_corner_points(self.box_dim,box_origin_pos.tolist(),box_origin_orn_eul)

    surface_1 = [box_corners_world[2],box_corners_world[3],box_corners_world[6],box_corners_world[7]]
    surface_2 = [box_corners_world[3],box_corners_world[0],box_corners_world[7],box_corners_world[4]]
    surface_3 = [box_corners_world[0],box_corners_world[1],box_corners_world[4],box_corners_world[5]]
    surface_4 = [box_corners_world[1],box_corners_world[2],box_corners_world[5],box_corners_world[6]]

    if self.sur_flag==1:
      surface=surface_1
    elif self.sur_flag==2:
      surface=surface_2
    elif self.sur_flag==3:
      surface=surface_3
    elif self.sur_flag==4:
      surface=surface_4

    point_on_sur = self.get_point_on_surface(surface,u,v)
    
    current_position = current_position-2*step_size*normal
    # print(current_position)
    jointPoses = self.bullet_client.calculateInverseKinematics(self.panda,pandaEndEffectorIndex, current_position, orn, ll, ul,
    jr, rp, maxNumIterations=1000,residualThreshold=0.0001)

    for i in range(pandaNumDofs):
        self.bullet_client.resetJointState(self.panda, i, jointPoses[i]) 

    current_position = point_on_sur
    # print(current_position)
    jointPoses = self.bullet_client.calculateInverseKinematics(self.panda,pandaEndEffectorIndex, current_position, orn, ll, ul,
    jr, rp, maxNumIterations=1000,residualThreshold=0.0001)

    for i in range(pandaNumDofs):
        self.bullet_client.resetJointState(self.panda, i, jointPoses[i])
    
    return current_position
  

  def get_u(self,orn):

    if orn >= 0.5*math.pi/6 and orn < 0.5*math.pi/4:
      u = 0.1
    elif orn >= 0.5*math.pi/12 and orn < 0.5*math.pi/6:
      u = 0.3
    elif orn > -0.5*math.pi/12 and orn < 0.5*math.pi/12:
      u = 0.5
    elif orn >= -0.5*math.pi/6 and orn <= -0.5*math.pi/12:
      u = 0.7
    elif orn > -0.5*math.pi/4 and orn  < -0.5*math.pi/6:
      u = 0.9
    elif orn > 0.5*math.pi/4:
      u = 0.1
    elif orn < -0.5*math.pi/4:
      u = 0.9

    return u 


  def update_v_rewards(self,v,number_of_step):
    
    # for relative dist
    z_before = self.box_centroid_list[number_of_step-2][2]
    z_after = self.box_centroid_list[number_of_step-1][2]

    z_tilt_before = self.point_on_tilting_axis_list[number_of_step-2][2]
    z_tilt_after = self.point_on_tilting_axis_list[number_of_step-1][2]

    if self.sur_flag==1:
      r_rel_dist = ((z_after-z_before)/self.step_size)

      r_rel_dist_mod = ((z_tilt_after-z_tilt_before)/self.step_size)
    else:
      r_rel_dist = (-(z_after-z_before)/self.step_size)
      r_rel_dist_mod = (-(z_tilt_after-z_tilt_before)/self.step_size)


    # for toppling
    x_orn_before = self.box_orientation_list[number_of_step-2][0]
    x_orn_initial = self.box_orn_euler[0]
    x_orn_after = self.box_orientation_list[number_of_step-1][0]

    # r_toppling = 1 if (x_orn_after-x_orn_before) > 0.005 else 0
    r_toppling = 1 if abs(x_orn_after-x_orn_initial) > 0.005 else 0
    r_toppling = abs(x_orn_after-x_orn_initial)/math.pi

    # no movement
    distance = np.linalg.norm(self.box_centroid_list[number_of_step-2] - self.box_centroid_list[number_of_step-1])
    r_no_move = 1 if distance < 0.0005 else 0

    # print(r_rel_dist,r_rel_dist_mod,r_toppling,r_no_move,r_no_move,distance)

    alpha,beta,gamma = 1,20,1


    # print(z_before,z_after)

    
      # self.v_rewards[v]+=1
      # self.v_rewards[v]+=((z_after-z_before)/self.step_size)
    # reward=(alpha*r_rel_dist - beta*r_toppling - gamma*r_no_move)
    reward=(alpha*r_rel_dist_mod - beta*r_toppling - gamma*r_no_move)
    self.v_rewards_sum[v] += reward
    self.v_rewards_count[v] += 1

    self.v_rewards[v] = self.v_rewards_sum[v] / self.v_rewards_count[v]
    # self.v_rewards[v] = self.v_rewards_sum[v]
    


  # def get_v(self,number_of_step):
  #   if number_of_step > self.play_phase_start_idx and number_of_step <= self.play_phase_start_idx + self.play_phase_length:
  #     v = self.v_list[0]
  #     if self.sequence_index==0 and number_of_step > self.play_phase_start_idx+2:
  #       self.update_v_rewards(v,number_of_step)
  #   elif number_of_step > self.play_phase_start_idx + self.play_phase_length and number_of_step <= self.play_phase_start_idx + 2*self.play_phase_length:
  #     v = self.v_list[1]
  #     if self.sequence_index==0 and number_of_step > self.play_phase_start_idx + self.play_phase_length+2:
  #       self.update_v_rewards(v,number_of_step)
  #   elif number_of_step > self.play_phase_start_idx + 2*self.play_phase_length and number_of_step <= self.play_phase_start_idx + 3*self.play_phase_length:
  #     v = self.v_list[2]
  #     if self.sequence_index==0 and number_of_step > self.play_phase_start_idx + 2*self.play_phase_length+2:
  #       self.update_v_rewards(v,number_of_step)
  #   elif number_of_step> self.play_phase_start_idx + 3*self.play_phase_length:
  #     # self.tot_rewards_for_v = [
  #     #   np.sum(self.rewards_list[self.rewards_list_play_idx_change[i]:self.rewards_list_play_idx_change[i]+18])
  #     #   for i in range(3)]
      
  #     # max_idx = np.argmax(self.tot_rewards_for_v)
  #     # v = self.v_list[max_idx]

  #     self.percentage_changes = [
  #       abs(self.rewards_list[self.rewards_list_play_idx_change[i] + (self.play_phase_length-2) - 1] - 
  #       self.rewards_list[self.rewards_list_play_idx_change[i]]) /
  #       max(abs(self.rewards_list[self.rewards_list_play_idx_change[i]]), 1e-6) * 100  
  #       for i in range(3)
  #       ]
  #     print(self.rewards_list[self.rewards_list_play_idx_change[0]],self.rewards_list[self.rewards_list_play_idx_change[0] + (self.play_phase_length-2) - 1]) 
  #     print(self.rewards_list[self.rewards_list_play_idx_change[1]],self.rewards_list[self.rewards_list_play_idx_change[1] + (self.play_phase_length-2) - 1]) 
  #     print(self.rewards_list[self.rewards_list_play_idx_change[2]],self.rewards_list[self.rewards_list_play_idx_change[2] + (self.play_phase_length-2) - 1]) 

  #     max_idx = np.argmax(self.percentage_changes)  
  #     v = self.v_list[max_idx]  

      
  #   else:
  #     v = 0.5

  #   return v
  
  def get_v(self, number_of_step):
    play_length = self.play_phase_length

    if number_of_step > self.play_phase_over_idx:
        best_v = max(self.v_rewards, key=self.v_rewards.get)
        if self.current_v!=best_v:
          self.current_v=best_v
          self.sequence_index=1
        return best_v

    # Initialize current_v if it's the first time
    if self.current_v_phase_start is None and number_of_step>self.play_phase_start_idx:
        self.current_v_phase_start = number_of_step
        print('Here is v_phase_start',self.current_v_phase_start)

    # Check if current v phase is over, select a new v using UCB

    if number_of_step>self.play_phase_start_idx:
      if  (number_of_step - self.current_v_phase_start) >= play_length:
          total_counts = sum(self.v_rewards_count[v] for v in self.v_list) + 1  # avoid log(0)
          c = 1.0  # UCB exploration factor

          # Calculate UCB score for each v
          ucb_scores = {}
          for v in self.v_list:
              count = self.v_rewards_count[v] + 1e-5  # Avoid division by zero
              avg_reward = self.v_rewards[v]
              ucb_scores[v] = avg_reward + c * math.sqrt(math.log(total_counts) / count)

          #FOR SEQUENTIALLY MAX
          selected_v = max(ucb_scores, key=ucb_scores.get) 

          #FOR RANDOM TIE BREAK
          max_ucb = max(ucb_scores.values())
          candidates = [v for v, score in ucb_scores.items() if score == max_ucb]
          selected_v = random.choice(candidates)

          # Update tracking variables
          if self.current_v!=selected_v:
            self.current_v=selected_v
            print('here v = ',selected_v)
            
            print(f'rewards list index start = {len(self.rewards_list)} ')
            self.rewards_list_play_idx_change.append(len(self.rewards_list))
            print(self.rewards_list_play_idx_change)
            self.sequence_index=1


          self.current_v = selected_v
          self.current_v_phase_start = number_of_step
          

          print(f"[UCB] Step {number_of_step} | New v selected: {selected_v:.3f} | UCB: {ucb_scores}")

      # Update reward for current v only after step 2 of the phase
      if self.sequence_index == 0 and (number_of_step - self.current_v_phase_start) > 1:
          self.update_v_rewards(self.current_v, number_of_step)

    return self.current_v


  def find_orn_bin_index(self,orn, bins):
    for index, (lower, upper) in enumerate(bins):
        if lower <= orn < upper:
            return index
    # Return -1 if `orn` is outside all bins
    return -1

  def get_visual_reward(self, obj_bb, roi,bottom_offset=0):

    start_row_obj = obj_bb[0]
    end_row_obj = obj_bb[1]
    start_col_obj = obj_bb[2]
    end_col_obj = obj_bb[3]

    start_row = roi[0]
    end_row = roi[1]
    start_col = roi[2]
    end_col = roi[3]

    center_col = (end_col-start_col)/2
    center_to_end = end_col-center_col

    hor_dist = min((end_col-start_col_obj),(end_col_obj-start_col))
    ver_dist = end_row-bottom_offset-end_row_obj
    
    living_reward  = -10
    hor_wt = 0.75
    ver_wt = 0.25

    reward = (center_to_end-(hor_wt*hor_dist)-(ver_wt*ver_dist)+living_reward)*100/center_to_end
    print('*****************')
    print(reward,center_to_end,hor_dist)
    print('*****************')

    return reward


  def extract_intrinsics_from_projection_matrix(self,proj_matrix, width, height):
    proj = np.array(proj_matrix).reshape((4, 4), order='F')
    fx = proj[0, 0] * width / 2
    fy = proj[1, 1] * height / 2
    cx = width / 2
    cy = height / 2
    return fx, fy, cx, cy
  
  def convert_depth_buffer_to_meters(self,depth_buffer, near=0.01, far=10.0):
    # depth_buffer = np.array(depth_buffer)
    return (far * near) / (far - (far - near) * depth_buffer)
  
  def depth_to_point_cloud(self,depth_image, mask, proj_matrix, view_matrix, width, height):
    fx, fy, cx, cy = self.extract_intrinsics_from_projection_matrix(proj_matrix, width, height)
    rows, cols = np.where(mask)
    depth_values = depth_image[rows, cols]
    # print(depth_values)

    # Camera space
    x = (cols - cx) * depth_values / fx
    y = (rows - cy) * depth_values / fy
    z = depth_values
    points_camera = np.stack([x, y, z], axis=1)

    # print(points_camera.shape)

    # Convert to world space
    view_mat = np.array(view_matrix).reshape(4, 4, order='F')
    cam_to_world = np.linalg.inv(view_mat)

    points_world = []
    for pt in points_camera:
        pt_hom = np.array([*pt, 1.0])
        world_pt = cam_to_world @ pt_hom
        points_world.append(world_pt[:3])

    points_world = np.array(points_world)

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_world)

    return points_world, pcd
  

  def estimate_tilt(self,points_world):
    pca = PCA(n_components=3)
    pca.fit(points_world)

    normal = pca.components_[2]  # normal to base plane
    tilt_angle = np.arccos(np.clip(np.dot(normal, [0, 1, 0]), -1.0, 1.0))
    tilt_angle_deg = np.degrees(tilt_angle)
    return tilt_angle_deg, normal


  def estimate_base_center(self,points_world):
    lowest_z = np.percentile(points_world[:, 2], 5)  # Bottom 5% of Z-values
    base_points = points_world[points_world[:, 2] < lowest_z + 0.01]
    return np.mean(base_points, axis=0)


  def visualize_point_cloud(self,pcd, normal=None, center=None):
    geometries = [pcd]

    if center is not None and normal is not None:
        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.01,
            cone_radius=0.02,
            cylinder_height=0.2,
            cone_height=0.04
        )
        arrow.paint_uniform_color([1.0, 0.0, 0.0])

        z_axis = np.array([0, 0, 1])
        axis = np.cross(z_axis, normal)
        angle = np.arccos(np.clip(np.dot(z_axis, normal), -1.0, 1.0))
        if np.linalg.norm(axis) > 1e-6:
            R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis / np.linalg.norm(axis) * angle)
            arrow.rotate(R, center=False)
        arrow.translate(center)
        geometries.append(arrow)

    o3d.visualization.draw_geometries(geometries)


  def step(self,total_steps):

    self.i+=1 #STEP
    print(f'------------------ STEP: {self.i}-------------------')
    
    width, height, rgbImage, depthBuffer, segmentationMask = self.bullet_client.getCameraImage(
        320, 200, flags=self.bullet_client.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX
    )

    # print(width,height)
   
    target_object_id = self.box

    #####################################################################################################
    ##########################################VISUAL REWARDS#############################################

    # depthBuffer = np.array(depthBuffer).reshape((height, width))
    # segmentationMask = np.array(segmentationMask).reshape((height, width))

    # depth_image = self.convert_depth_buffer_to_meters(depthBuffer)
    # print(depth_image)

    # points_world, pcd = self.depth_to_point_cloud(depth_image, segmentationMask, self.projection_matrix, self.view_matrix, width, height)
    # # print(points_world,pcd)

    # tilt_angle, normal = self.estimate_tilt(points_world)
    # base_center = self.estimate_base_center(points_world)

    # print(tilt_angle,base_center)

    # # self.visualize_point_cloud(pcd)
    # # time.sleep(1000)

    # # # if self.i>1:
    # depthBuffer = np.array(depthBuffer).reshape((height, width))
    # segmentationMask = np.array(segmentationMask).reshape((height, width))  
    
    # object_depth_values = depthBuffer[segmentationMask == target_object_id]

    # # # max_depth = np.max(object_depth_values)
    # # # min_depth = np.min(object_depth_values)
    # # # mean_depth = np.mean(object_depth_values)
    # # # print(max_depth,min_depth,mean_depth)

    # object_pixels = np.where(segmentationMask == target_object_id)
    # object_pixel_rows = object_pixels[0]
    # object_pixel_cols = object_pixels[1]

    # # (row,col) tuple
    # object_pixels_list = list(zip(object_pixel_rows,object_pixel_cols))
    # # print(object_pixels_list)


    # # print(object_pixels)
    # # print(object_pixel_rows)
    # # print(object_pixel_cols)
    # pixel_depth_info = list(zip(object_pixel_rows, object_pixel_cols, object_depth_values))
    # # print(pixel_depth_info)

    # # object_pixel_count = np.sum(segmentationMask == target_object_id)
    # # print(object_pixel_count)


    # # Define the region of interest (ROI) bounds
    # start_row, end_row = 0, height  
    # start_col, end_col = int(width*0.), int(width*1)  
    
    # roi_segmentation_mask = segmentationMask[start_row:end_row, start_col:end_col]
    
    # # object_pixel_count_in_roi = np.sum(roi_segmentation_mask == target_object_id)
    # # print(f"Number of pixels where object {target_object_id} is present in the ROI: {object_pixel_count_in_roi}")


    # roi = [start_row,end_row,0,end_col]

    # # Get the indices of pixels where the target object is present in the ROI
    # object_pixels_in_roi = np.where(roi_segmentation_mask == target_object_id)
    # # print(object_pixels_in_roi)

    # if object_pixels_in_roi[0].size > 0:  # Check if the object exists in the ROI
        
    #     start_row_obj = np.min(object_pixels_in_roi[0])  
    #     end_row_obj = np.max(object_pixels_in_roi[0])    
    #     start_col_obj = np.min(object_pixels_in_roi[1])  
    #     end_col_obj = np.max(object_pixels_in_roi[1])    

    #     print(f"Bounding Box of object {target_object_id} in ROI:")
    #     print(f"Start Row: {start_row_obj}, End Row: {end_row_obj}, Start Col: {start_col_obj}, End Col: {end_col_obj}")
    #     print(end_col)

    #     obj_bb = [start_row_obj,end_row_obj,start_col_obj,end_col_obj]

    #     visual_reward = self.get_visual_reward(obj_bb,roi)
    #     self.visual_reward_list.append(visual_reward)



    # else:
    #     print(f"Object {target_object_id} not found in the ROI.")
    #     center_col = (end_col-start_col)/2
    #     visual_reward = 100
    #     self.visual_reward_list.append(visual_reward)
        
    ########################################################################################################  
    ########################################################################################################
    ########################################################################################################

    # self.bullet_client.getCameraImage(320,200)
    self.bullet_client.configureDebugVisualizer(self.bullet_client.COV_ENABLE_SINGLE_STEP_RENDERING,1,rgbBackground=[1, 1, 1])

    box_position, box_orientation = self.get_box_position()
    box_orientation=self.bullet_client.getEulerFromQuaternion(box_orientation)

    # print(box_position,n1,n2,n3)
    # print(box_orientation)

    box_origin,_ = self.get_box_origin()
    centroid = np.array(box_position)
    self.box_centroid_list.append(centroid)
    self.box_orientation_list.append(box_orientation)
    self.box_origin_list.append(box_origin)

    # print(f'box_pos : ', centroid)
    
    angle = box_orientation[1]
    # print(angle)
    self.thetas.append(angle)

    side_length = self.box_dim[0]

    vertices = self.square_vertices_xz_plane(centroid, side_length, angle)
    normals = self.unit_normals_to_centroid_xz(vertices, centroid) # THESE NORMALS ARE TOWARDS THE SURFACE

    if self.sur_flag==1:
       normal = normals[0]
       normal_perp = self.rotate_vector_90_deg_y(normal)
      #  print(normal,normal_perp)
    elif self.sur_flag==2:
       normal = normals[3]
       normal_perp = self.rotate_vector_90_deg_y(normal)
    elif self.sur_flag ==3:
       normal = normals[2]
       normal_perp = self.rotate_vector_90_deg_y(normal)

    orn = self.target_orientation

    euler_orn = self.bullet_client.getEulerFromQuaternion(orn)
    orn_euler = [euler_orn[0],angle,euler_orn[2]]
    orn = self.bullet_client.getQuaternionFromEuler(orn_euler)

    end_effector_state = self.bullet_client.getLinkState(self.panda, pandaEndEffectorIndex)
    current_position = end_effector_state[0]
    # print(f'{current_position}')
    step_size = self.step_size
    # step_size = 0.2


    if self.policy==0:
       normal = self.normal
       normal_perp = self.rotate_vector_90_deg_y(normal)

    # point_on_tilting_axis = box_origin+normal*((self.box_dim[1])/2)
    # self.point_on_tilting_axis_list.append(point_on_tilting_axis)

    tipping_point = self.get_tipping_point_center(self.box_dim, box_position, box_orientation)
    self.point_on_tilting_axis_list.append(tipping_point)
    # print(box_origin)

    # print(tipping_point)

    # print(box_position)

    target_position = current_position+step_size*normal #FOR 200 STEPS and step_size = 0.05
    # print(f'{target_position}')
    # target_position = current_position+0.1*self.normal #FOR 20 STEPS
    # target_position = current_position+1.*self.normal


    ############ MAIN ACTION #####################################
    if self.i<=total_steps and self.sequence_index==0:

      jointPoses = self.bullet_client.calculateInverseKinematics(self.panda,pandaEndEffectorIndex, target_position, orn, ll, ul,
        jr, rp, maxNumIterations=1000,residualThreshold=0.0001)
      
      
      for i in range(pandaNumDofs):
        #self.bullet_client.setJointMotorControl2(self.panda, i, self.bullet_client.POSITION_CONTROL, jointPoses[i],force=5 * 240.)
        self.bullet_client.setJointMotorControl2(self.panda, i, self.bullet_client.POSITION_CONTROL, jointPoses[i], force = self.forces[self.force_index])

      # print(f'MOVED')
      end_effector_state = self.bullet_client.getLinkState(self.panda, pandaEndEffectorIndex)
      current_position = end_effector_state[0]
      # print(f'{end_effector_state[0]}')
      reward = self.reward_function()
      # print('Reward = ',reward)
      self.rewards_list.append(reward)

    
    ##################################### PLAY PHASE ###############################
    # elif self.sequence_index==1 and self.i<=self.play_phase_start_idx+len(self.v_list)*self.play_phase_length+2:
    elif self.sequence_index==1 and self.i>= self.play_phase_start_idx and self.i<self.play_phase_over_idx:

      current_position = current_position-2*step_size*normal
      jointPoses = self.bullet_client.calculateInverseKinematics(self.panda,pandaEndEffectorIndex, current_position, orn, ll, ul,
      jr, rp, maxNumIterations=1000,residualThreshold=0.0001)

      for i in range(pandaNumDofs):
        self.bullet_client.setJointMotorControl2(self.panda, i, self.bullet_client.POSITION_CONTROL, jointPoses[i], force = self.forces[self.force_index])

      print('HERE PHASE 1 DONE')
      
      self.sequence_index = 2

    ##################################### PLAY PHASE ###############################
    elif self.sequence_index == 2 and self.i<=self.play_phase_start_idx+len(self.v_list)*self.play_phase_length+3:
      box_origin_pos, box_origin_orn_eul = self.get_box_origin()
      box_corners_world = self.get_box_corner_points(self.box_dim,box_origin_pos.tolist(),box_origin_orn_eul)

      surface_1 = [box_corners_world[2],box_corners_world[3],box_corners_world[6],box_corners_world[7]]
      surface_2 = [box_corners_world[3],box_corners_world[0],box_corners_world[7],box_corners_world[4]]
      surface_3 = [box_corners_world[0],box_corners_world[1],box_corners_world[4],box_corners_world[5]]
      surface_4 = [box_corners_world[1],box_corners_world[2],box_corners_world[5],box_corners_world[6]]

      if self.sur_flag==1:
        surface=surface_1
      elif self.sur_flag==2:
        surface=surface_2
      elif self.sur_flag==3:
        surface=surface_3
      elif self.sur_flag==4:
        surface=surface_4
 

      v = self.get_v(self.i)
      

      print('v = ',v)
      u = 0.5
      point_on_sur = self.get_point_on_surface(surface,u,v)
      
      # print(current_position)
      jointPoses = self.bullet_client.calculateInverseKinematics(self.panda,pandaEndEffectorIndex, point_on_sur, orn, ll, ul,
      jr, rp, maxNumIterations=1000,residualThreshold=0.0001)

      # for i in range(pandaNumDofs):
      #   self.bullet_client.setJointMotorControl2(self.panda, i, self.bullet_client.POSITION_CONTROL, jointPoses[i], force = self.forces[self.force_index])

      for i in range(pandaNumDofs):
        self.bullet_client.resetJointState(self.panda, i, jointPoses[i])

      print('HERE PHASE 2 DONE')
      self.sequence_index=0
      

    v = self.get_v(self.i)
    print('v = ',v)
    # print(self.play_phase_start_idx)

    # if self.current_v!=v:
    #   self.current_v=v
    #   print('here v = ',v)
      
    #   print(f'rewards list index start = {len(self.rewards_list)} ')
    #   self.rewards_list_play_idx_change.append(len(self.rewards_list))
    #   print(self.rewards_list_play_idx_change)
    #   self.sequence_index=1

    

    ############################# PUSH PHASE #########################################

    if self.i> self.play_phase_start_idx+len(self.v_list)*self.play_phase_length+3:
      if self.sequence_index==1:

        current_position = current_position-2*step_size*normal
        jointPoses = self.bullet_client.calculateInverseKinematics(self.panda,pandaEndEffectorIndex, current_position, orn, ll, ul,
        jr, rp, maxNumIterations=1000,residualThreshold=0.0001)

        for i in range(pandaNumDofs):
          self.bullet_client.setJointMotorControl2(self.panda, i, self.bullet_client.POSITION_CONTROL, jointPoses[i], force = self.forces[self.force_index])

        self.sequence_index = 2

      # # if self.sequence_index == 2:

      # #   current_position = current_position-2*step_size*normal_perp
      # #   jointPoses = self.bullet_client.calculateInverseKinematics(self.panda,pandaEndEffectorIndex, current_position, orn, ll, ul,
      # #   jr, rp, maxNumIterations=1000,residualThreshold=0.0001)

      # #   for i in range(pandaNumDofs):
      # #     self.bullet_client.setJointMotorControl2(self.panda, i, self.bullet_client.POSITION_CONTROL, jointPoses[i], force = self.forces[self.force_index])

      # #   self.sequence_index = 3




      if self.sequence_index == 2:
        box_origin_pos, box_origin_orn_eul = self.get_box_origin()
        box_corners_world = self.get_box_corner_points(self.box_dim,box_origin_pos.tolist(),box_origin_orn_eul)

        surface_1 = [box_corners_world[2],box_corners_world[3],box_corners_world[6],box_corners_world[7]]
        surface_2 = [box_corners_world[3],box_corners_world[0],box_corners_world[7],box_corners_world[4]]
        surface_3 = [box_corners_world[0],box_corners_world[1],box_corners_world[4],box_corners_world[5]]
        surface_4 = [box_corners_world[1],box_corners_world[2],box_corners_world[5],box_corners_world[6]]

        if self.sur_flag==1:
          surface=surface_1
        elif self.sur_flag==2:
          surface=surface_2
        elif self.sur_flag==3:
          surface=surface_3
        elif self.sur_flag==4:
          surface=surface_4

        u = self.get_u(angle)
        print('u = ',u)
        v = self.current_v
        point_on_sur = self.get_point_on_surface(surface,u,v)
        
        # print(current_position)
        jointPoses = self.bullet_client.calculateInverseKinematics(self.panda,pandaEndEffectorIndex, point_on_sur, orn, ll, ul,
        jr, rp, maxNumIterations=1000,residualThreshold=0.0001)

        # for i in range(pandaNumDofs):
        #   self.bullet_client.setJointMotorControl2(self.panda, i, self.bullet_client.POSITION_CONTROL, jointPoses[i], force = self.forces[self.force_index])

        for i in range(pandaNumDofs):
          self.bullet_client.resetJointState(self.panda, i, jointPoses[i])

        self.sequence_index=0

      


        
      check_orn_bin = self.find_orn_bin_index(angle,self.orn_bins)

      if check_orn_bin!=self.orn_bin_index:
        print(check_orn_bin)
        self.orn_bin_index = check_orn_bin
        self.sequence_index = 1
        
        self.target_orientation = orn
      
    contact_points = self.bullet_client.getContactPoints(bodyA=self.panda, bodyB=self.box)
    # If there are any contact points, it means a collision is occurring
    if len(contact_points) > 0:
        # print("Collision detected between Panda and Box!")
        max_cnt_force = -math.inf
        for point in contact_points:
          force = point[9]  # Normal force magnitude
          # print(f"Contact Force: {force}")
          if force>max_cnt_force:
            max_cnt_force=force

        # print(f'Max Contact Force  = {max_cnt_force}')
        self.contact_force.append(max_cnt_force)


        if self.thomp_start==False:
          self.play_phase_start_idx = self.i
          self.play_phase_over_idx = self.play_phase_start_idx + len(self.v_list) * self.play_phase_length

          print('play start = ', self.play_phase_start_idx)
          print('play end = ', self.play_phase_over_idx)
          self.thomp_start=True

        percent_change_reward = abs(self.rewards_list[-1]-self.rewards_list[-2])*100/abs(self.rewards_list[-2])
        # percent_change_reward = abs(self.visual_reward_list[-1]-self.visual_reward_list[-3])*100/abs(self.visual_reward_list[-3])
        # print(f'Change = {percent_change_reward} % ')

        # end_effector_state = self.bullet_client.getLinkState(self.panda, pandaEndEffectorIndex)
        # current_position = end_effector_state[0]
        # u = self.get_u(angle)
        # print(angle,u)
        # current_position = self.move_to_point_on_box(current_position,orn,normal,step_size,u,0.5)

        if percent_change_reward < 1:
           self.force_index = min(self.force_index+1,len(self.forces)-1)
           self.force_counter+=1

            
             
          #  random_number = np.random.rand()
          #  print(random_number)
          #  end_effector_state = self.bullet_client.getLinkState(self.panda, pandaEndEffectorIndex)
          #  current_position = end_effector_state[0]
        
          #  current_position = self.move_to_point_on_box(current_position,orn,normal,step_size,random_number,0.5)

          
           if self.force_counter<=len(self.forces):
            print('!!! changed !!! at step =  ', self.i)


    self.forces_list.append(self.forces[self.force_index])
    print(f'-------------------------------------')
    
  
