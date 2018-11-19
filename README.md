# sound_source_localization
A ROS package that contains a CNN-based sound source localizer.
This package contains a script that creates a dataset with
two robots (currently specialized for Toyota HSR but can be easily adapted to other robots) that have microphone arrays and speaker.

## Requirements (TODO: should be declared in package.xml)
* ROS kinetic
* rosbridge
* audio_capture (apt install ros-kinetic-audio-capture)
* pyaudio (apt install python-pyaudio)
* scipy
* tensorflow

## Usage
### Data recording
Assuming there are two robots connected to the same network, and each of them has its own roscore.
On robot A (or a terminal in which ROS_MASTER_URI specifies robot A),
``` 
roslaunch sound_source_localization record.launch \
local_robot_hostname:=robot_a.local remote_robot_hostname:=robot_b.local \
role_name:=A dataset_name:=foo
```
On robot B (or a terminal in which ROS_MASTER_URI specifies robot B),
``` 
roslaunch sound_source_localization record.launch \
local_robot_hostname:=robot_b.local remote_robot_hostname:=robot_a.local \
role_name:=B dataset_name:=foo
```
See PACKAGE_DIR/launch/record.launch for details on other parameters.
Recording process will be started after both commands are invoked.
Acquired data will be saved in PACKAGE_DIR/data/foo (in cases where dataset_name:=foo).

### Training
```
roslaunch sound_source_localization train.launch dataset_name:=foo model_name:=bar
```
Dataset will be read from PACKAGE_DIR/data/foo, and trained model will be saved in PACKAGE_DIR/data/bar.
The names of ckpt files will look like 'model-1000.data-00000-of-00001'.

### Test
```
roslaunch sound_source_localization test.launch dataset_name:=foo model_name:=bar
```
When model_name is specified as an existing directory (e.g. 'bar'), model parameters will be restored from the latest ckpt in PACKAGE_DIR/data/bar.
It can also be like 'bar/model-1000' to specify the training step.

### Prediction
```
roslaunch sound_source_localization predict.launch robot_hostname:=robot.local model_name:=bar
```
The ckpt file can be specified in the same manner as Test.
