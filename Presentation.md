## 01 Overview and inputs

### What does the panning module do?

- It **generates the path and speed profiles** for the autonomous vehicle to follow.
- It **takes inputs** from the **perception**, **localization**, and **prediction modules**. It uses them to **plan a safe**, **smooth**, and **efficient trajectory.**
- Its main **goal** is to ensure that the **AV moves** from point A to point B while **avoiding obstacles** and **adhering to traffic rules.**

##### Perception

Provides data on the _surrounding environment_, such as obstacles, lane lines, and road boundaries..

##### Prediction

_Anticipates_ the future _movements_ of _dynamic objects_, such as cars or pedestrians.

##### Localization

Gives the _exact position of the vehicle_ relative to the road and surroundings.

##### Routing

Provides the _high-level route from the current location_ to the destination, which the Planning Module refines into a detailed path.

## 02 Core Components

#### Reference Line

A **baseline** path for the vehicle to follow, typically derived from the **routing module.**

![Italian Trulli](http://localhost:8000/views/presentations/images/reference-line.png)

#### Path and Speed Planning

![Italian Trulli](http://localhost:8000/views/presentations/images/path-speed-planning-1.png) ![Italian Trulli](http://localhost:8000/views/presentations/images/path-speed-planning-2.png)

Generates the **trajectory** (combination of the path and speed profile) that the vehicle will follow.

### Scenario-Based Planning

![Italian Trulli](http://localhost:8000/views/presentations/images/scenario-based-planning.png)

Adapts **planning strategies to different driving scenarios**, such as highway driving, urban environments, and parking.
## 03 Key algorithms

### EM Planner

**Optimizes** both the **path and speed** by considering the **reference line**, the surrounding environment, obstacles, and driving conditions. It ensures that the **trajectory** generated is **smooth and feasible** for the vehicle to follow.

### Hybrid A*

- Combines the **classic A* search** algorithm with **vehicle kinematic models** to generate feasible paths in complex environments. (tight-space navigation)
- Searches through a grid or continuous space to **find an optimal path**, considering the vehicle's **motion constraints**.
- Used for parking and complex maneuvers (e.g., U-turns, reverse parking).

### QP-Spline-Path Optimizer

- **Quadratic Programming** (QP) combined with **spline optimization** to create **smooth paths** by minimizing a function subject to linear constraints.
- Ensures to **avoid** minimal **deviation from the reference line**, **abrupt changes in direction**, **speed**, or **acceleration**.
#### Introduction

The **QP-Spline-Path Optimizer** combines **quadratic programming (QP)** and **spline interpolation** to generate smooth and optimized paths for autonomous vehicles. It operates in a **station-lateral coordinate system** where the path is divided into segments and modeled as splines.

#### Key Concepts:

1. **Objective Function**:
    
    - The path is split into **n spline segments**, each represented by a polynomial. The objective function aims to minimize the overall cost, which accounts for the first, second, and third derivatives of the spline (i.e., smoothness). This is converted into a QP formulation.
2. **QP Formulation**:
    
    - The cost function is expressed as a quadratic optimization problem:
        - Minimize: 12⋅xT⋅H⋅x+fT⋅x\frac{1}{2} \cdot x^T \cdot H \cdot x + f^T \cdot x21​⋅xT⋅H⋅x+fT⋅x
        - Subject to linear equality and inequality constraints for the spline coefficients.
3. **Constraints**:
    
    - **Initial and End Point Constraints**: These ensure that the path starts and ends at specific locations with predefined offsets and derivatives. This is represented as a system of linear equations.
    - **Joint Smoothness Constraints**: Ensure that consecutive spline segments are continuous and smooth at the connection points, including up to third-order derivatives.
    - **Boundary Constraints**: Sample points along the path and ensure that the generated trajectory stays within certain boundaries (e.g., road width, obstacle avoidance). These are converted into inequality constraints.

#### Conclusion

The QP-Spline-Path Optimizer provides an effective way to generate smooth paths by solving a QP problem over spline segments. It ensures the trajectory is continuous, smooth, and safe by incorporating constraints that handle the vehicle’s initial and final positions, joint smoothness, and boundary limits.

### Open Space Planner

The **Open Space Planner Algorithm** is a scenario-specific planning tool within Apollo, designed for complex maneuvers such as reverse parking and sharp U-turns. It takes input from **perception data** (like obstacles) and a **Region of Interest (ROI)** from HD maps. The algorithm follows two stages:

1. **Searching-Based Planning**: A raw trajectory is generated using a vehicle kinematic model (Hybrid A*), which plots an initial path for the ego vehicle.
2. **Optimization**: The trajectory is refined for smoothness and collision avoidance, ensuring better riding comfort and easier tracking by the control module. The optimized path adjusts the spacing of trajectory points based on turns and straight paths, effectively controlling the vehicle’s speed and movement.

The output is then sent to the control module for execution. Currently, the Open Space Planner is used for valet parking and curb-side parking (including parallel parking). Future applications may involve more complex maneuvers, such as tight U-turns and emergency curb-side parking.

#### References

[1]: Dolgov, Dmitri, et al. "Path Planning for Autonomous Vehicles in Unknown Semi-Structured Environments." The International Journal of Robotics Research, vol. 29, no. 5, 2010, pp. 485-501., doi:10.1177/0278364909359210.

[2]: Prodan, Ionela, et al. "Mixed-integer representations in control design: Mathematical foundations and applications."" Springer, 2015.

[3]: Xiaojing Zhang, et al. "Optimization-Based Collision Avoidance" (arXiv:1711.03449).

#### Introduction

The **Open Space Trajectory Optimizer** refines initial trajectories for open space scenarios like parking, using various optimization algorithms to generate smooth and feasible paths.

#### Code Structure

The optimizer starts with inputs like the initial trajectory, target points, boundaries, and obstacle data. It preprocesses the data, checks for invalid cases, and then generates a **coarse trajectory** using the Hybrid A* algorithm.

#### Optimization Process

The optimization can follow two paths:

1. **Without Parallel Smoothing**: The coarse trajectory is smoothed by solving a distance-based approach using Eigen matrices for state and control variables.
2. **With Parallel Smoothing**: The trajectory is segmented, optimized in parts, and then combined for a final smooth path.
### Generate Final Trajectory Summary

#### Introduction
The **Generate Final Trajectory** section is responsible for producing the final trajectory in open space scenarios (like parking) by utilizing the **Open Space Trajectory Provider**. This module manages the flow and calls the **Hybrid A* and trajectory smoothing algorithms** to create smooth, feasible paths.

#### Code Structure
The **OpenSpaceTrajectoryProvider::Process()** function is called during the **VALET_PARKING** stage. It ensures safety by generating a stop trajectory during the "park and go" check stage. A thread is initiated for planning the first trajectory and monitors trajectory states (updated, skipped, error).

#### Key Processes
1. **Vehicle Stop Due to Fallback**:
   - The **IsVehicleStopDueToFallBack()** function checks if the vehicle has stopped due to a fallback by analyzing vehicle speed and acceleration.

2. **Trajectory Stitching**:
   - If the vehicle is stopped or replanning is required, **ComputeStitchingTrajectory()** or **ComputeReinitStitchingTrajectory()** will generate a new trajectory based on the vehicle's state and previous trajectory.

#### Replanning Conditions
The trajectory is replanned in several cases, such as when the vehicle is near its destination, the previous planning failed, or trajectory errors persist for more than 10 seconds. The final trajectory is optimized and adjusted for smooth transitions.

#### Output
The final output is a smooth, optimized trajectory ready to be executed by the vehicle.

### Algorithm Detail
The algorithm focuses on handling fallback situations, replanning trajectories when necessary, and ensuring a seamless transition between the previous and new trajectories by stitching them together based on the vehicle's state and environmental conditions.

#### Output

The optimized trajectory is then converted to world coordinates and loaded into the system, ready for execution by the control module. The algorithm ensures smooth transitions and control of the vehicle throughout the process.

### Algorithm Details

The optimizer transforms the trajectory into matrix form for optimization, handling parameters like position, orientation, speed, and steering, ensuring the vehicle follows a smooth and feasible path.

### TNT (Trajectory and Target Prediction)

### Inter-TNT Evaluator Summary

The **Inter-TNT** model, introduced in Apollo 7.0, improves trajectory prediction for obstacles by utilizing a **VectorNet encoder** and a **TNT decoder**. It interacts with the latest autonomous vehicle (AV) planning trajectory to predict short-term trajectories for surrounding obstacles. This model improves performance over previous versions, reducing inference time and enhancing accuracy.

### Key Components:

- **VectorNet (Encoder)**: Encodes the trajectories of the AV and obstacles as polylines (sequential points) and contextual map features from HD maps. The data is processed using subgraph and global graph networks.
- **TNT (Decoder)**: Generates multiple target points and potential trajectories for obstacles. It uses a scoring system to select the most likely trajectories.
- **Interaction Module**: Measures the interaction between the AV and obstacles by comparing position and velocity differences, improving the accuracy of trajectory predictions.

### Performance Improvements:

Inter-TNT improves prediction accuracy by over 20% in terms of minADE and minFDE, while reducing inference time from 15 ms to 10 ms.

### Output:

The model generates multi-modal trajectories and assigns confidence scores, allowing Apollo to make better-informed decisions about the behavior of surrounding obstacles.

### Graph-based Planning (VectorNet)

**Encodes the input features** (such as obstacle trajectories and map elements) into a vectorized format, which can then be processed by the planning algorithms


## 04 Module Interaction

#### 1. **Prediction Module**:

- **Role**: The **Prediction Module** provides future trajectory predictions of obstacles around the vehicle (e.g., pedestrians, cyclists, other vehicles). These predictions include the possible paths and behaviors of the obstacles.
- **Interaction**: The Planning Module uses this data to adjust the autonomous vehicle's (AV) trajectory dynamically. By anticipating where obstacles might move, the Planning Module can make real-time decisions to avoid collisions and ensure smooth driving.
- **Example**: If the Prediction Module foresees a pedestrian crossing the street ahead, the Planning Module will reroute or adjust speed to safely avoid the pedestrian.

#### 2. **Perception Module**:

- **Role**: The **Perception Module** detects and classifies obstacles in the environment (e.g., cars, people, traffic signs) and continuously updates the AV's understanding of its surroundings.
- **Interaction**: The Planning Module relies on the Perception Module for up-to-date information on the environment. This real-time perception data informs the planner about obstacles, road conditions, and traffic rules, allowing the planner to generate a safe trajectory.
- **Example**: If a new obstacle, like a stopped car, appears in the vehicle's path, the Perception Module detects it, and the Planning Module recalculates the trajectory to avoid the obstacle.

#### 3. **Routing Module**:

- **Role**: The **Routing Module** provides a high-level path or "route" from the AV's current location to its destination, similar to GPS navigation. This route serves as the reference for more detailed planning by the Planning Module.
- **Interaction**: The Planning Module breaks down this high-level route into a detailed, feasible trajectory by considering dynamic obstacles, road boundaries, and other real-time factors. The Routing Module provides the general direction, while the Planning Module handles the fine-grained decisions.
- **Example**: The Routing Module may suggest a path that requires turning at an intersection, but the Planning Module will decide the exact timing, lane selection, and speed adjustments to safely complete the turn while avoiding vehicles and pedestrians.

#### 4. **Control Module**:

- **Role**: The **Control Module** takes the trajectory generated by the Planning Module and converts it into physical commands for the vehicle, such as steering, throttle, and braking.
- **Interaction**: The Control Module executes the planned trajectory by controlling the vehicle’s movement. The Planning Module provides precise instructions for how the vehicle should behave (e.g., maintain a certain speed, turn at a certain angle), and the Control Module ensures those instructions are carried out.
- **Example**: If the Planning Module generates a trajectory that requires slowing down due to upcoming traffic, the Control Module will reduce the vehicle’s speed smoothly in response.

#### 5. **Localization Module**:

- **Role**: The **Localization Module** determines the AV’s precise location within the environment by combining GPS, IMU, LiDAR, and other sensor data.
- **Interaction**: The Planning Module requires accurate localization to ensure that the generated trajectory corresponds correctly to the AV's actual position. If the vehicle's position shifts due to localization updates, the Planning Module recalculates the trajectory to maintain accuracy.
- **Example**: If the Localization Module detects that the vehicle is drifting slightly off-center in its lane, the Planning Module adjusts the trajectory to bring it back to the center.

## 05 Scenario-Based Planning

#### **Overview**
- **Scenario-Based Planning** allows Apollo’s Planning Module to adapt its trajectory generation based on specific driving environments and situations. Each scenario presents different challenges and requires different strategies for safe and efficient autonomous driving. Apollo’s approach ensures the vehicle behaves appropriately in diverse conditions, such as city driving, highway cruising, or parking.

#### **How It Works**
- **Scenario Detection**: The system continuously monitors the vehicle's environment and state to identify which scenario is currently in play. This could include driving on a highway, navigating an intersection, or executing a parking maneuver. Each scenario has distinct requirements and constraints, which are used to guide the planning process.
  
- **Scenario-Specific Algorithms**: Once a scenario is detected, Apollo uses specialized algorithms designed to handle that particular situation. For instance:
  - **Lane Following**: In typical highway or city driving scenarios, the Planning Module uses algorithms to maintain the vehicle within its lane, considering nearby vehicles, speed limits, and lane changes.
  - **Intersection Handling**: When approaching an intersection, the Planning Module will focus on traffic signals, cross traffic, and pedestrian movement. It will optimize for safe passage while minimizing wait times.
  - **Parking**: The system uses the **Open Space Planner** for complex maneuvers such as reverse parking or parallel parking in tight spaces. This algorithm uses the Hybrid A* and trajectory optimization techniques for precise control.
  - **Merging and Lane Changes**: In highway scenarios, the system must account for high-speed merges and lane changes, ensuring the vehicle can safely adjust its position relative to fast-moving vehicles.

#### **Benefits of Scenario-Based Planning**
- **Adaptability**: The Planning Module can dynamically adjust its strategies based on the current scenario. For example, the system will plan differently when driving in a clear lane versus when navigating a dense urban environment filled with obstacles.
  
- **Efficiency**: By selecting the optimal strategy for each scenario, Apollo can ensure smoother, safer driving, reducing unnecessary stops and enhancing traffic flow.
  
- **Safety**: Scenario-based planning allows the system to focus on the most critical aspects of the current environment. For example, in a parking scenario, the focus will shift to low-speed maneuverability and obstacle avoidance rather than maintaining high-speed efficiency.

#### **Example Scenarios**
1. **Urban Driving**:
   - **Challenges**: Tight streets, frequent stops, pedestrian crossings, dynamic obstacles (e.g., cyclists, parked cars).
   - **Scenario-Specific Behavior**: Slow speed, frequent stops, heightened focus on pedestrian detection, and narrow obstacle avoidance.

2. **Highway Driving**:
   - **Challenges**: High speeds, lane changes, merging traffic, maintaining a safe following distance.
   - **Scenario-Specific Behavior**: Smooth lane keeping, adjusting speed based on traffic, and anticipating merging vehicles.

3. **Parking**:
   - **Challenges**: Limited space, precise maneuvering, and low-speed control.
   - **Scenario-Specific Behavior**: The Open Space Planner is activated for intricate parking maneuvers, optimizing for accuracy and safety in tight spaces.

4. **Intersections**:
   - **Challenges**: Managing cross traffic, traffic lights, pedestrian movements.
   - **Scenario-Specific Behavior**: Careful timing of movement through the intersection, considering right-of-way, traffic signals, and pedestrian safety.

#### **Scenario Management**
- Apollo uses a **Scenario Manager** that tracks the vehicle’s state and environmental factors to decide which scenario to activate. This manager seamlessly switches between scenarios as the vehicle moves through different driving contexts.

### **Talking Points for the Presentation**:
- Emphasize how **scenario-based planning improves adaptability and safety** by tailoring the vehicle’s behavior to the situation at hand.
- Provide examples of specific scenarios the system handles and how the planning strategies differ.
- Discuss the **seamless transition** between scenarios, ensuring the vehicle can adapt to changing environments in real-time.
  
This section will help your audience understand how scenario-based planning contributes to Apollo's ability to navigate complex, real-world environments safely and efficiently.


## 06 Where is the AI?

Certainly! Here’s a breakdown of the **models used in perception and prediction** within Apollo, explaining their role, the function they serve in the broader framework, and the pipeline that brings everything together.

### **Perception Module**

The **Perception Module** is responsible for understanding the environment by detecting and classifying objects such as vehicles, pedestrians, traffic signs, and lane markings. This module relies heavily on **deep learning (DL) models** to process data from sensors like cameras, LiDAR, and radar.

#### **Models Used in Perception**:
1. **Convolutional Neural Networks (CNNs)**:
   - **Function**: CNNs are widely used for image-based tasks such as object detection, classification, and segmentation. These models take camera input and detect vehicles, pedestrians, traffic lights, road signs, and more.
   - **Example**: For camera-based perception, Apollo uses CNN-based models like **YOLO** (You Only Look Once) or **Faster R-CNN** for real-time object detection and segmentation.
   
2. **PointNet and PointPillars** (for LiDAR data):
   - **Function**: These DL models process point clouds generated by LiDAR sensors, helping detect objects, their shapes, and positions in 3D space.
   - **Example**: **PointNet** and **PointPillars** are popular choices in Apollo’s LiDAR processing pipeline. PointNet directly works on point cloud data, while PointPillars converts 3D point clouds into pseudo-images for faster processing.

3. **Multi-Sensor Fusion Models**:
   - **Function**: Combining data from multiple sensors (like cameras, LiDAR, and radar) allows the perception system to create a more robust and accurate understanding of the environment. DL models fuse the data to identify objects more reliably, particularly in challenging conditions (e.g., low light, occlusion).
   - **Example**: Apollo uses models that fuse data from LiDAR and cameras to reduce false positives and enhance the accuracy of object detection.

#### **Pipeline for Perception**:
1. **Data Acquisition**: The sensors (camera, LiDAR, radar) capture data about the environment.
2. **Preprocessing**: The raw data is preprocessed to normalize it, filter out noise, and prepare it for analysis (e.g., image resizing for CNNs or point cloud normalization for PointNet).
3. **Object Detection**:
   - Camera feeds are processed using CNNs to identify objects (cars, pedestrians, etc.).
   - LiDAR point clouds are processed by models like PointPillars to detect 3D objects.
4. **Object Classification and Tracking**: Detected objects are classified into categories (e.g., pedestrian, cyclist, vehicle), and the system keeps track of their movement across frames (tracking algorithms).
5. **Output**: The perception module outputs a list of detected objects, their classifications, and their positions and velocities. This data is passed to the Prediction and Planning Modules.

### **Prediction Module**

The **Prediction Module** predicts the future positions and behaviors of dynamic objects detected by the perception system (e.g., vehicles, pedestrians). The goal is to anticipate the trajectories of these objects over a short time horizon to allow the AV to plan accordingly.

#### **Models Used in Prediction**:
1. **VectorNet (Graph Neural Network)**:
   - **Function**: VectorNet is a **graph neural network (GNN)** used to encode the relationship between map elements and the detected objects. It models the interaction between the AV, surrounding vehicles, pedestrians, and the road environment.
   - **How it Works**: VectorNet takes vectorized data, like object trajectories and HD map information, and encodes them into a graph. Each node represents a detected object, and the edges represent their interactions.
   
2. **TNT (Target-Driven Trajectory Prediction)**:
   - **Function**: TNT (Target-driven Network) generates multiple possible future trajectories for each object and assigns probabilities to each. It’s a **multi-modal prediction model** that considers various potential actions that surrounding vehicles or pedestrians might take.
   - **How it Works**: TNT generates multiple target points around the detected object, then calculates likely trajectories to those targets. It assigns probabilities to each trajectory based on past behaviors and interaction with the AV's planned path.
   
3. **Inter-TNT (Joint VectorNet-TNT-Interaction)**:
   - **Function**: **Inter-TNT** combines **VectorNet** and **TNT** while incorporating interaction with the AV's planned trajectory. This model accounts for how surrounding obstacles might react to the AV’s movements, leading to more realistic trajectory predictions.
   - **How it Works**: Inter-TNT uses VectorNet as the encoder and TNT as the decoder. It processes object trajectories as polylines and generates multiple possible futures based on interactions between objects and the AV. The model re-weights trajectory predictions by considering how the AV's movements could influence obstacle behavior.

#### **Pipeline for Prediction**:
1. **Data Input**: The Prediction Module receives detected objects (with positions, velocities, and classifications) from the Perception Module.
2. **Interaction Encoding**:
   - **VectorNet** encodes the relationship between the objects and the map elements (e.g., lanes, intersections). The model processes each object’s trajectory as a polyline and encodes interactions between objects.
3. **Multi-Modal Trajectory Prediction**:
   - **TNT** generates multiple possible trajectories for each object. These trajectories reflect different potential behaviors (e.g., stopping, accelerating, turning).
4. **Interaction with the AV**:
   - **Inter-TNT** adds an interaction mechanism, measuring how the AV’s planned trajectory will influence the future behavior of nearby objects. This step adjusts the predicted trajectories accordingly.
5. **Output**: The Prediction Module outputs multiple potential trajectories for each object, along with a probability for each trajectory. These predictions are passed to the Planning Module to assist in generating a safe and optimal path.

### **Role in the Framework**
- **Perception**: The perception models are responsible for understanding the current state of the environment by detecting and classifying objects around the AV. This is crucial for enabling the AV to make informed decisions.
- **Prediction**: The prediction models provide foresight, helping the AV anticipate what surrounding vehicles and pedestrians might do next. This allows the Planning Module to consider possible future interactions and generate a trajectory that avoids collisions.

### **Overall Pipeline**:
1. **Perception**: Sensors capture the environment. DL models (e.g., CNNs, PointNet) process the data to detect objects and classify them.
2. **Prediction**: The detected objects are fed into the Prediction Module, where GNN and TNT-based models predict the future positions and behaviors of those objects, considering interactions with the AV.
3. **Planning**: The predicted trajectories from the Prediction Module inform the Planning Module, allowing it to generate a safe, optimized trajectory for the AV.



## TNT

### **NT (Target-Driven Trajectory Prediction) Architecture**

The **TNT (Target-driven Trajectory Prediction)** architecture is designed to predict the future trajectories of dynamic obstacles (like vehicles, pedestrians, etc.) surrounding the autonomous vehicle (AV). TNT is a **multi-modal prediction model**, meaning it can predict multiple possible future trajectories for each obstacle, rather than just a single one. This helps the AV plan around uncertainty in the environment.

Here's a detailed explanation of the architecture:

### **Key Components of TNT Architecture**

1. **Encoder (VectorNet or Similar Models)**:
    
    - **Function**: The encoder processes the input data, which includes the obstacle trajectories, the map information, and interactions between objects. It transforms these inputs into a feature representation that can be used by the decoder to generate predictions.
    - **How It Works**: The input trajectories are represented as **polylines** (i.e., sequential points that form the paths of obstacles), and features such as speed, heading, and position are extracted. In Apollo, **VectorNet** is typically used as the encoder, which operates on graph-like data structures to capture the relationships between objects and the road network.
2. **Target Prediction (Target-Driven Prediction)**:
    
    - **Function**: This is a key part of TNT. Instead of predicting one future trajectory, the model first predicts **target points** where the obstacle might end up after a certain time. These target points represent potential goals or destinations for the dynamic object.
    - **How It Works**: The model generates **N candidate target points** around the obstacle. These points are potential future positions, considering various possibilities like the obstacle continuing straight, turning left, stopping, etc. The targets are typically sampled uniformly in the obstacle’s environment, covering different plausible outcomes.
3. **Trajectory Prediction (Multi-Modal Prediction)**:
    
    - **Function**: After identifying the potential target points, the model generates **M trajectories** for each target. Each trajectory represents a possible path the obstacle could take to reach one of the predicted target points.
    - **How It Works**: For each obstacle, TNT generates a set of possible trajectories leading to each of the target points. These trajectories consider the obstacle’s current velocity, acceleration, and surrounding environment (such as road lanes and traffic). The trajectories are often parameterized by time and consider the dynamics of the obstacle (e.g., a car’s turning radius, speed limits).
4. **Scoring and Selection**:
    
    - **Function**: The model assigns **likelihood scores** to each of the predicted trajectories. These scores indicate how likely it is that the obstacle will follow a given trajectory.
    - **How It Works**: The scoring mechanism evaluates each trajectory based on factors such as historical behavior, interaction with other objects, and road constraints. The model produces a probability distribution over the possible trajectories, and the trajectories with the highest scores are considered the most likely.
5. **Interaction with AV’s Trajectory** (in Inter-TNT variant):
    
    - **Function**: In more advanced implementations like **Inter-TNT**, the predicted trajectories of surrounding obstacles are influenced by the AV’s planned trajectory. This accounts for the fact that other objects might change their behavior in response to the AV’s movement.
    - **How It Works**: After generating the initial predictions, the model updates the trajectories based on the AV’s planned path. For example, if the AV is predicted to stop at a crosswalk, nearby pedestrians may be more likely to cross. This interaction mechanism adjusts the trajectory predictions accordingly.

### **Pipeline of TNT Architecture**

1. **Input**:
    
    - The input includes the **obstacle trajectories** (i.e., the current positions and velocities of the surrounding vehicles, pedestrians, etc.) and the **HD map information** (i.e., road layout, lane boundaries, etc.).
    - The input is encoded as polylines representing the paths of each obstacle.
2. **Feature Encoding**:
    
    - The encoder, often **VectorNet**, processes the input features to create a high-level representation of the obstacle’s current state and its relation to the surrounding environment. This encoded information is passed to the next stage.
3. **Target Prediction**:
    
    - The model samples multiple possible **target points** in the environment where the obstacle might end up. These targets could represent different plausible future positions, such as the object moving straight, turning, or stopping.
4. **Trajectory Prediction**:
    
    - For each target point, the model generates multiple possible trajectories the obstacle might take to reach that target. These trajectories are influenced by the obstacle’s current state (speed, direction) and the constraints imposed by the environment (lanes, intersections, etc.).
5. **Scoring and Selection**:
    
    - The generated trajectories are scored based on their likelihood. The model assigns probabilities to each trajectory, indicating how likely the obstacle is to follow that specific path. These probabilities are based on factors like historical motion patterns and interactions with other obstacles.
6. **Interaction with AV (if Inter-TNT)**:
    
    - In **Inter-TNT**, after the initial predictions are made, the model takes into account the planned trajectory of the AV. It recalculates the likelihood of each trajectory based on how the obstacle might respond to the AV’s movements.
7. **Output**:
    
    - The final output is a set of **multi-modal predicted trajectories** for each obstacle, with associated probabilities. This information is passed to the **Planning Module**, where the AV’s trajectory is adjusted based on the predicted movements of nearby objects.

### **Benefits of TNT Architecture**

- **Multi-Modal Predictions**: TNT’s ability to generate multiple possible trajectories allows the system to account for uncertainty in obstacle behavior, which is essential for safe autonomous driving.
- **Target-Driven Approach**: By first predicting target points and then generating trajectories, the model simplifies the complex task of trajectory prediction into smaller, more manageable steps.
- **Interaction Awareness**: The Inter-TNT variant adds a level of sophistication by incorporating the AV’s trajectory into the predictions, making the system more adaptive to real-time driving dynamics.

### **Role in Apollo**

- **TNT** is a critical part of Apollo’s **Prediction Module**, responsible for forecasting the movements of dynamic obstacles around the AV. These predictions are vital inputs to the **Planning Module**, allowing it to generate safe and efficient paths for the AV while avoiding potential collisions with other vehicles and pedestrians.


### **Neural Network Backbone**

#### **1. VectorNet (GNN Backbone for TNT)**

- **Backbone**: The backbone for **VectorNet** is a **Graph Neural Network (GNN)**, which is designed to handle graph-structured data like roads, lanes, and obstacle trajectories represented as polylines. The network processes these polylines by first encoding each one locally (subgraph) and then encoding the global context (global graph network).
    
- **Why GNN?**: GNNs are well-suited for this task because they can model the relationships between objects in a structured environment, such as the connectivity between lanes and obstacles in a driving scenario. GNNs allow the model to capture both local (e.g., vehicle-to-lane) and global (e.g., vehicle-to-vehicle) interactions.
    
- **Input Data**:
    
    - **Polylines** representing the trajectories of objects: Each polyline consists of sequential points in space, such as (x, y) coordinates.
    - **Attributes**: Each polyline point may include additional features such as velocity, acceleration, and heading direction.
    - **Map Information**: High-definition (HD) maps represented as polylines for lanes, roads, and intersections.
- **Output Data**:
    
    - **Encoded Features**: The GNN outputs a high-level feature vector for each object and its interaction with the environment, which is passed to the trajectory prediction stage (TNT).

#### **2. TNT Decoder (Multi-Modal Prediction using Deep Networks)**

- **Backbone**: The **TNT decoder** typically uses **fully connected neural networks (MLPs)** and other dense layers to handle the prediction of multiple potential trajectories based on the encoded features from VectorNet.
    
- **Input Data**:
    
    - **Encoded Features** from VectorNet: These represent the object's state (position, velocity, etc.) and its interaction with the environment (lanes, roads, and other vehicles).
    - **Target Points**: The model generates **N target points** around the object. These target points are locations where the object might end up, given its current trajectory.
- **Output Data**:
    
    - **Predicted Trajectories**: The TNT decoder outputs **M trajectories** for each target point, each representing a possible path that the obstacle might take.
    - **Likelihood Scores**: For each predicted trajectory, the model provides a probability score, indicating how likely the obstacle is to follow that specific trajectory.

### **Input Data Format**

1. **Trajectory Polylines** (for objects):
    
    - Each trajectory is a sequence of points representing the motion of an obstacle over time. For example:
        - **Polyline Format**: `[(x1, y1, t1), (x2, y2, t2), ... (xn, yn, tn)]`
        - **Attributes**: Each point can include additional information like speed, acceleration, heading, and type of object (car, pedestrian, etc.).
2. **Map Data**:
    
    - **Polylines** representing the road, lane boundaries, intersections, etc.
    - **Attributes**: Lane types, speed limits, road curvature, etc.
3. **Object Features**:
    
    - Features like **position**, **velocity**, **acceleration**, and **heading direction** for each detected object are extracted from the perception system. These are inputs to the GNN for encoding.
4. **Interaction Information**:
    
    - For **Inter-TNT**, the model also takes into account the AV’s own trajectory as part of the input to adjust obstacle predictions based on expected interactions with the AV.

### **Output Data Format**

1. **Predicted Target Points**:
    
    - **N target points** are generated for each obstacle. These points are locations the obstacle might reach at the end of the prediction horizon (e.g., 3-5 seconds into the future).
    - **Target Format**: `(x, y)` coordinates with associated probabilities.
2. **Predicted Trajectories**:
    
    - **M trajectories** are generated for each target point, each representing a possible path that the obstacle might take to reach that target.
    - **Trajectory Format**: Each trajectory is a sequence of points similar to the input trajectory polylines, but now representing a future prediction.
        - Example: `[(x1_pred, y1_pred, t1), (x2_pred, y2_pred, t2), ... (xn_pred, yn_pred, tn)]`
3. **Likelihood Scores**:
    
    - Each predicted trajectory is associated with a likelihood score (between 0 and 1), indicating the probability of that trajectory being the correct one based on the object’s current behavior and interaction with the AV.

### **How the Input and Output Work in the Pipeline**

1. **Perception Input**: The perception system detects the obstacles around the AV and generates a list of polylines representing their past and current trajectories.
    
2. **GNN Encoding (VectorNet)**:
    
    - The polylines (trajectories) and map information are passed into VectorNet, where they are processed by the GNN. This process creates feature embeddings that capture both the local object dynamics and the global context of the environment.
3. **Target Generation (TNT Decoder)**:
    
    - Based on the encoded features, the TNT decoder generates **N target points** around each object. These target points represent possible future positions the obstacle might occupy.
4. **Trajectory Prediction**:
    
    - For each target point, the decoder generates **M possible trajectories**, representing different ways the object could reach that target. The model considers the object’s current velocity, direction, and possible reactions to the AV’s trajectory.
5. **Scoring and Selection**:
    
    - The model assigns a probability score to each predicted trajectory, indicating the likelihood that the object will follow that path. The trajectories with the highest scores are considered the most likely.
6. **Output to Planning**:
    
    - The final output is a set of **multi-modal predicted trajectories**, each with associated likelihoods. This data is passed to the **Planning Module**, which uses it to generate the AV’s trajectory while avoiding predicted obstacles.

### **Conclusion**

The TNT architecture relies heavily on **Graph Neural Networks (VectorNet)** for encoding spatial and temporal relationships between objects and map features. The **TNT decoder** then takes these encoded features and generates multiple potential future trajectories for each object. This approach allows the system to handle the inherent uncertainty in dynamic environments by considering multiple possible outcomes and helping the autonomous vehicle navigate safely.


### **Neural Network Backbone**

#### **1. VectorNet (GNN Backbone for TNT)**

- **Backbone**: The backbone for **VectorNet** is a **Graph Neural Network (GNN)**, which is designed to handle graph-structured data like roads, lanes, and obstacle trajectories represented as polylines. The network processes these polylines by first encoding each one locally (subgraph) and then encoding the global context (global graph network).
    
- **Why GNN?**: GNNs are well-suited for this task because they can model the relationships between objects in a structured environment, such as the connectivity between lanes and obstacles in a driving scenario. GNNs allow the model to capture both local (e.g., vehicle-to-lane) and global (e.g., vehicle-to-vehicle) interactions.
    
- **Input Data**:
    
    - **Polylines** representing the trajectories of objects: Each polyline consists of sequential points in space, such as (x, y) coordinates.
    - **Attributes**: Each polyline point may include additional features such as velocity, acceleration, and heading direction.
    - **Map Information**: High-definition (HD) maps represented as polylines for lanes, roads, and intersections.
- **Output Data**:
    
    - **Encoded Features**: The GNN outputs a high-level feature vector for each object and its interaction with the environment, which is passed to the trajectory prediction stage (TNT).

#### **2. TNT Decoder (Multi-Modal Prediction using Deep Networks)**

- **Backbone**: The **TNT decoder** typically uses **fully connected neural networks (MLPs)** and other dense layers to handle the prediction of multiple potential trajectories based on the encoded features from VectorNet.
    
- **Input Data**:
    
    - **Encoded Features** from VectorNet: These represent the object's state (position, velocity, etc.) and its interaction with the environment (lanes, roads, and other vehicles).
    - **Target Points**: The model generates **N target points** around the object. These target points are locations where the object might end up, given its current trajectory.
- **Output Data**:
    
    - **Predicted Trajectories**: The TNT decoder outputs **M trajectories** for each target point, each representing a possible path that the obstacle might take.
    - **Likelihood Scores**: For each predicted trajectory, the model provides a probability score, indicating how likely the obstacle is to follow that specific trajectory.

### **Input Data Format**

1. **Trajectory Polylines** (for objects):
    
    - Each trajectory is a sequence of points representing the motion of an obstacle over time. For example:
        - **Polyline Format**: `[(x1, y1, t1), (x2, y2, t2), ... (xn, yn, tn)]`
        - **Attributes**: Each point can include additional information like speed, acceleration, heading, and type of object (car, pedestrian, etc.).
2. **Map Data**:
    
    - **Polylines** representing the road, lane boundaries, intersections, etc.
    - **Attributes**: Lane types, speed limits, road curvature, etc.
3. **Object Features**:
    
    - Features like **position**, **velocity**, **acceleration**, and **heading direction** for each detected object are extracted from the perception system. These are inputs to the GNN for encoding.
4. **Interaction Information**:
    
    - For **Inter-TNT**, the model also takes into account the AV’s own trajectory as part of the input to adjust obstacle predictions based on expected interactions with the AV.

### **Output Data Format**

1. **Predicted Target Points**:
    
    - **N target points** are generated for each obstacle. These points are locations the obstacle might reach at the end of the prediction horizon (e.g., 3-5 seconds into the future).
    - **Target Format**: `(x, y)` coordinates with associated probabilities.
2. **Predicted Trajectories**:
    
    - **M trajectories** are generated for each target point, each representing a possible path that the obstacle might take to reach that target.
    - **Trajectory Format**: Each trajectory is a sequence of points similar to the input trajectory polylines, but now representing a future prediction.
        - Example: `[(x1_pred, y1_pred, t1), (x2_pred, y2_pred, t2), ... (xn_pred, yn_pred, tn)]`
3. **Likelihood Scores**:
    
    - Each predicted trajectory is associated with a likelihood score (between 0 and 1), indicating the probability of that trajectory being the correct one based on the object’s current behavior and interaction with the AV.

### **How the Input and Output Work in the Pipeline**

1. **Perception Input**: The perception system detects the obstacles around the AV and generates a list of polylines representing their past and current trajectories.
    
2. **GNN Encoding (VectorNet)**:
    
    - The polylines (trajectories) and map information are passed into VectorNet, where they are processed by the GNN. This process creates feature embeddings that capture both the local object dynamics and the global context of the environment.
3. **Target Generation (TNT Decoder)**:
    
    - Based on the encoded features, the TNT decoder generates **N target points** around each object. These target points represent possible future positions the obstacle might occupy.
4. **Trajectory Prediction**:
    
    - For each target point, the decoder generates **M possible trajectories**, representing different ways the object could reach that target. The model considers the object’s current velocity, direction, and possible reactions to the AV’s trajectory.
5. **Scoring and Selection**:
    
    - The model assigns a probability score to each predicted trajectory, indicating the likelihood that the object will follow that path. The trajectories with the highest scores are considered the most likely.
6. **Output to Planning**:
    
    - The final output is a set of **multi-modal predicted trajectories**, each with associated likelihoods. This data is passed to the **Planning Module**, which uses it to generate the AV’s trajectory while avoiding predicted obstacles.

### **Conclusion**

The TNT architecture relies heavily on **Graph Neural Networks (VectorNet)** for encoding spatial and temporal relationships between objects and map features. The **TNT decoder** then takes these encoded features and generates multiple potential future trajectories for each object. This approach allows the system to handle the inherent uncertainty in dynamic environments by considering multiple possible outcomes and helping the autonomous vehicle navigate safely.
