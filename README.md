# Reward-penalty-Dice-loss

The paper of this project, named ***Learning Non-Unique Segmentation with Reward-Penalty Dice Loss***, was accepted by WCCI (IJCNN) 2020. Most research and applications of semantic segmentation focus on addressing unique segmentation problems, where there is only one gold standard segmentation result for every input image. This may not be true in some problems, e.g., medical applications. We may have non-unique segmentation annotations as different surgeons may perform successful surgeries for the same patient in slightly different
ways. To comprehensively learn non-unique segmentation tasks, we propose the reward-penalty Dice loss (RPDL) function as the optimization objective for deep convolutional neural networks (DCNN). RPDL is capable of helping DCNN learn non-unique segmentation by enhancing common regions and penalizing outside ones.

## Prerequisites
Python 3.6, Tensorflow 1.14.0, Keras 2.2.4

## Baselines
* Weighted cross-entropy loss (WCEL)
* Dice loss (DL)

## Published dataset
* Cortical mastoidectomy (CM) dataset in folder [cm_data](cm_data)

## Implement
Here is an example of 3D U-net.

### Experiment 1
U-net with either WCEL or DL can be trained in [train_dice_wce.py](train_dice_wce.py).

### Experiment 2
U-net with RPDL can be trained in [train_rpdice.py](train_rpdice.py).

### Pre-trained models
Pre-trained models are in [model_bone](model_bone) and [model_surgeon](model_surgeon).

## Notes

When we train our models with different loss functions, we should either implement them in Experiment 1 or 2, with details in our paper. In experiment 1, different bones are picked out for testing. In experiment 2, different surgeons are picked out for testing.  

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
