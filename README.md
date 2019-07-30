# Joint Measurement of Localization and Detection of Sound Events

This repository implements the sound event localization and detection (SELD) metrics proposed in 

> Annamaria Mesaros, Sharath Adavanne, Archontis Politis, Toni Heittola, and Tuomas Virtanen, "Joint Measurement of Localization and Detection of Sound Events" in IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA 2019)

For the evaluation of the metrics, we use the recently proposed [SELDnet method](https://github.com/sharathadavanne/seld-net), and [TAU Spatial Sound Events 2019 - Microphone Array dataset](https://arxiv.org/abs/1905.08546).

If you are using this code or the datasets in any format, then please consider citing us.


## Setup
1. Clone the repository and update the submodules

   ```command
   git clone https://github.com/sharathadavanne/seld-metric.git
   cd seld-metric
   git submodule init
   git submodule update
   ```
   
2. Download the reference SELD labels - https://zenodo.org/record/2599196/files/metadata_dev.zip  Update the corresponding folder location in the [code](https://github.com/sharathadavanne/seld-metric/blob/7b0b49dd23f09019d80e503605d0d350df342272/test_metrics.py#L142).
3. Download the predicted SELD labels of the SELDnet method - https://zenodo.org/record/3354709
   This package consists of the SELDnet results at 5, 25 and 75 epochs. Update the location of the corresponding results folder in the [code](https://github.com/sharathadavanne/seld-metric/blob/7b0b49dd23f09019d80e503605d0d350df342272/test_metrics.py#L143).
4. Choose the segment length for evaluation in the [code](https://github.com/sharathadavanne/seld-metric/blob/7b0b49dd23f09019d80e503605d0d350df342272/test_metrics.py#L153).
5. Run the code

     ```python
        python test_metrics.py
     ```
  
Thats it! The results obtained should be identical to that in the paper.


## Additional details
- The `test_metrics.py` script is only used for testing and providing an example usage of the metrics defined in `SELD_evaluation_metrics.py` script. Ideally, you can use `SELD_evaluation_metrics.py` as a standalone script.
- The `SELD_evaluation_metrics.py` script has been tested on both python 2.x and 3.x
- The `test_metrics.py` in the current configuration only works for python 3.x, inorder to run it in python 2.x you will have to figure out a right way to import libraries as described in the [code here](https://github.com/sharathadavanne/seld-metric/blob/c0dc9dd23147cbb08ca6e2e6d2ee78ea7e3f5855/test_metrics.py#L22).

## Acknowledgement
This work has received funding from the European Research Council under the ERC Grant Agreement 637422 EVERYSOUND
