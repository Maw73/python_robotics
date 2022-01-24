# Forward kinematics

# Task
Write a function which will compute the forward kinematic task for a 6R manipulator.
After building the function, we can explore the range of the end-effector moving the first joint of the Motoman MA1400:
![Motoman MA1400](https://user-images.githubusercontent.com/98336978/150835030-8542d931-4cf0-4e4f-a45e-a76f32cfbd07.png)

# Function specifications:
Inputs of the function:
- mechanism - a python dictionary which will contain the Denavit-Hartenberg parameters
- joints - the theta angles (how will the robot move its joints)

Return value: 
- values of rotation and translation of the end-effector in the base coordinate system
