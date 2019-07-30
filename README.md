# Joint Measurement of Localization and Detection of Sound Events

This repository implements the sound event localization and detection (SELD) metrics proposed in 

> Annamaria Mesaros, Sharath Adavanne, Archontis Politis, Toni Heittola, and Tuomas Virtanen, "Joint Measurement of Localization and Detection of Sound Events" in IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA 2019)

For the evaluation of the metrics, we use the recently proposed [SELDnet method](https://arxiv.org/abs/1807.00129), and [TAU Spatial Sound Events 2019 - Microphone Array dataset](https://arxiv.org/abs/1905.08546).

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
  '''python
  python test_metrics.py
  '''
  
Thats it! The results obtained should be identical to that in the paper.
