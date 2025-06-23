import pybullet as p
import pybullet_data as pd
import math
import time
import numpy as np
import panda_sim as panda_sim
import matplotlib.pyplot as plt
import json

def extract_com_from_urdf(urdf_file_path):
	import xml.etree.ElementTree as ET

	# Load the URDF file
	urdf_file = urdf_file_path #URDF file path
	tree = ET.parse(urdf_file)
	root = tree.getroot()

	# Find the <origin> tag
	origin_tag = root.find(".//origin")

	if origin_tag is not None:
		xyz = origin_tag.get("xyz")
		if xyz:
			x, y, z = map(float, xyz.split())  # Extract x, y, z values
			return [x,y,z]
		else:
			print("No xyz attribute found in <origin> tag")
			return None
	else:
		print("No <origin> tag found in the URDF file")
		return None
 

def plot_arm_rewards(reward_dict, obj_com, box_height):
    # Sort dictionary by height (keys) to ensure correct spacing
    sorted_items = sorted(reward_dict.items())
    keys, values = zip(*sorted_items)  

    z_value = obj_com[2]
    v_com = round(z_value / box_height, 3)

    max_index = values.index(max(values))  
    max_key = keys[max_index]  
    max_value = values[max_index]  

    plt.figure(figsize=(8, 6))

    # Adjust bar width if necessary
    bar_width = 0.025  

    plt.barh(keys, values, height=bar_width, color='blue', alpha=0.7, label="v_rewards")

    # C.O.M horizontal line
    plt.axhline(y=v_com, color='red', linestyle='--', linewidth=1, label=f"C.O.M = {v_com}")

    # Highlight the max reward
    plt.scatter(max_value, max_key, color='green', s=150, label=f"Choosen V ({max_key})")

    plt.xlabel("Rewards")
    plt.ylabel("v (height)")
    plt.title("Rewards vs Height")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Ensure proper y-axis limits
    plt.ylim(min(keys) - 0.1, max(keys) + 0.1)  

    # Ensure evenly spaced y-ticks
    plt.yticks(np.arange(min(keys), max(keys) + 0.1, 0.1))  

    plt.show()

	
	


##################################################################################
urdf_file = "D:\Project\small_setup\small_setup\data\iitd_80cm_box\80cm_box.urdf" 
obj_com = extract_com_from_urdf(urdf_file)
##################################################################################


p.connect(p.GUI)
# p.connect(p.DIRECT)
p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP,1)
p.setAdditionalSearchPath(pd.getDataPath())

timeStep=1./60.
p.setTimeStep(timeStep)
p.setGravity(0,-9.8,0)

# v_list = [0.75,.7,0.6,0.5,0.4,0.3,0.25]
# v_list = [0.5]
v_list = [0.95,.9,.85,.8,.75,.7,0.65,0.6,0.55,0.5,0.45,0.4,0.35,0.3,0.25,0.2,0.15]
surface = 1
panda = panda_sim.PandaSim(p,[0,0,0],v_list,surface)
box_height = panda.box_dim[2]


# panda.v_list = [0.9,0.7,0.5,0.3,0.1]
# panda.v_list = [0.49,0.8,0.4,0.2]
# panda.v_rewards = {v: 0 for v in panda.v_list}

total_steps = 350
counter = 1


while (1):
	panda.step(total_steps)
	
	p.stepSimulation()
	time.sleep(timeStep)


	if counter==total_steps:
		break
	

	counter+=1

	
	# Check for keyboard events
	keyboard_events = p.getKeyboardEvents()
		
	# Check if the space bar (key code 32) is pressed
	if 32 in keyboard_events and keyboard_events[32] & p.KEY_WAS_RELEASED:
		print("Space bar pressed. Exiting...")
		break

#panda.save_images()


steps = list(range(len(panda.rewards_list)))


accumulated_rewards = np.cumsum(panda.rewards_list)

# print(panda.v_rewards_count)


plt.subplot(2, 1, 1)  # 2 rows, 1 column, first subplot
plt.plot(steps, panda.rewards_list,color='blue')
plt.xlabel('Steps')
plt.ylabel('Rewards')
plt.title('Rewards vs Steps')


plt.subplot(2, 1, 2)  # 2 rows, 1 column, second subplot
plt.plot(steps, accumulated_rewards,color='green')
plt.xlabel('Steps')
plt.ylabel('Accumulated Rewards')
plt.title('Accumulated Rewards vs Steps')
plt.tight_layout()
plt.show()

# vis_steps = list(range(len(panda.visual_reward_list)))
# accumulated_rewards = np.cumsum(panda.visual_reward_list)
# plt.subplot(2, 1, 1)  # 2 rows, 1 column, first subplot
# plt.plot(vis_steps, panda.visual_reward_list,color='blue')
# plt.xlabel('Steps')
# plt.ylabel('Visual Rewards')
# plt.title('Visual Rewards vs Steps')
# plt.subplot(2, 1, 2)  # 2 rows, 1 column, second subplot
# plt.plot(vis_steps, accumulated_rewards,color='green')
# plt.xlabel('Steps')
# plt.ylabel('Accumulated Visual Rewards')
# plt.title('Accumulated  VisualRewards vs Steps')
# plt.tight_layout()
# plt.show()

force_steps = list(range(len(panda.forces_list)))
plt.plot(force_steps, panda.forces_list,color='red')
plt.xlabel('Steps')
plt.ylabel('Force/Torque (N/Nm)')
plt.title('Force/Torque vs Steps')
plt.show()

force_steps = list(range(len(panda.contact_force)))
plt.plot(force_steps, panda.contact_force,color='green')
plt.xlabel('Steps')
plt.ylabel('Force/Torque (N/Nm)')
plt.title('Contact Force vs Steps')
plt.show()

# print(f'Pencentage change for v {panda.percentage_changes}')
print(len(panda.box_centroid_list))
print(panda.v_rewards)

plot_arm_rewards(panda.v_rewards,obj_com,box_height)

print('Simulation end')