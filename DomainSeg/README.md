## DomainSeg
Roadwork scenes and construction objects present a challenging edge-case scenario for self-driving cars. Roadwork objects can be placed dynamically and their position can be changed at a moments notice by construction workers. On many occassions, roadwork objects can be used to artifically narrow the driving corridor and guide vehicles to merge lanes. These scenarios are very demanding for self-driving cars, creating a need for a robust and reliable roadwork scene segmentation technology. DomainSeg addresses this key technology gap, delivering robust safety perception across urban driving scenarios, highways and even unstrcutured roads. It is able to adapt to challenging weather conditions such as snow and low-light, and is robust to edge cases such as detection of traffic cones that have been knocked over by other cars. DomainSeg is part of the [AutoSeg Foundation Model](../AutoSeg/README.md) which forms the core of the vision-pipeline of the [Autoware Privately Owned Vehicle Autonomous Highway Pilot System](..).

![SceneSeg GIF](../Media/DomainSeg_GIF.gif) ![SceneSeg GIF Rain](../Media/DomainSeg_GIF_2.gif)

During training, DomainSeg estimates a binary segmentation mask with a probability of each pixel belonging to a single 'super-class' of **roadwork objects**

## Watch the explainer video
Please click the video link to play - [***Video link***](https://drive.google.com/file/d/1gB7lIsvwm-4PdDg2fjSEBH0nqW0rI-4W/view?usp=sharing)

## Demo, Training, Inference, Visualization
Please see the [*Models*](https://github.com/autowarefoundation/autoware.privately-owned-vehicles/tree/main/Models) folder to access the pre-trained network weights for DomainSeg as well as scripts for network training, inference and visualization of network predictions.

## Performance Results
DomainSeg was trained using the ROADWork dataset by Carnegie Mellon University, yielding 7.25K training samples and 150 validation samples.

Mean Intersection Over Union (mIoU) scores for DomainSeg on validation samples are below

### Validation Set Performance - mIoU Scores
| |Validation | 
|--------|----|
| mIoU | **46.6** | 


### Inference Speed
Inference speed tests were performed on a laptop equipped with an RTX3060 Mobile Gaming GPU, and an AMD Ryzen 7 5800H CPU. The SceneSeg network comprises a total of 223.43 Billion Floating Point Operations.

#### FP32 Precision
At FP32 precision, SceneSeg achieved 14.3 Frames Per Second inference speed

#### FP16 Precision
At FP16 precision, SceneSeg achieved 23.9 Frames Per Second inference speed
