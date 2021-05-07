
# Crack_detection using FasterRCNN

### Problem Statement:
Cracks on the concrete surface are one of the earliest indications of degradation of the structure which is critical for the 
maintenance as well the continuous exposure will lead to the severe damage to the environment. 
Manual inspection is the acclaimed method for the crack inspection. In the manual inspection, the sketch of the crack is 
prepared manually, and the conditions of the irregularities are noted. Since the manual approach completely depends on the 
specialist’s knowledge and experience, it lacks objectivity in the quantitative analysis. 
So, automatic image-based crack detection is proposed as a replacement.

#### Configuration
* We are using **FasterRCNN** [TFOD API](https://github.com/tensorflow/models) to build this model.

# Installing
1. Clone the repository 
```shell
git clone https://github.com/rp926463-arch/Crack_detection.git
```
2.Go to folder path
```shell
cd $ROOT_DIR/fasterRCNN
```
3. Create conda virtual environment
```shell
conda create -n fasterRCNN python==3.6.9
conda activate fasterRCNN
```
4. Install prerequisites for Project
```shell
pip install -r requirements.txt
```
5. Run file
```shell
python app.py
```

# Testing
Upload image to webpage it will launch output using windows image viewer

#input & Output data present under data folder for reference.

![Output](https://github.com/rp926463-arch/Crack_detection/blob/main/data/crack_images_output/image4.PNG?raw=true)
