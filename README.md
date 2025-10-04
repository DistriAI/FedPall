# **Project Introduction**
Official repo of **FedPall: Prototype-based Adversarial and Collaborative Learning for Federated Learning with Feature Drift** , ICCV 2025.
***
In this paper, we propose the **FedPall** algorithm, which addresses the **feature drift** problem by using prototype-based adversarial collaborative learning.
***
![overview](./assets/overview.png)
# **Getting Started**
## **Dependencies**
This code requires as following
- conda 24.5.0
- python 3.12.5
- PyTorch 2.5.1
- Torchvision 0.20.1
- numpy 1.26.4
- pandas 2.2.2

You can configure the environment with the following command
```bash
conda install --yes --file requirements.txt
```
## Datasets
- Please download our datasets [here](https://drive.google.com/drive/folders/1xLxaz3zJRqZbTVDzkoAoWZiX50gwZI_4?usp=sharing), put under `./data/` directory.
- The directory structure of the data folder after the data is added is
```.
data
├── Digits
│   ├── SubDataset[MNIST | MNIST_M | SVHN | SynthDigits | USPS] // SubDataset
│   │   ├── train.pkl                                           // Full training set
│   │   └── test.pkl                                            // Full test set
│   │   ├── partitions
|   │   │   └── train_part[i:0-9].pkl                           // The training set subset extracted according to 10%
│   └── data_pre.py                                             // Data preprocessing
├── office_caltech_10
│   ├── SubDataset[amazon | caltech | dslr | webcam]            // original data
│   ├── SubDataset_train.pkl                                    // Full training set
│   └── SubDataset_test.pkl                                     // Full test set
│   └── data_pre.py                                             // Data preprocessing
└───PACS                                                        // The structure is similar to office-caltech-10
```
You can preprocess the raw data of the specified dataset using the following command.
```bash
cd data
cd Digits
python data_pre.py
```
## Usage
### Train
Please using following commands to train a model with federated learning strategy.
- **--mode** specify federated learning strategy, option: SingleSet | fedavg | fedprox | perfedavg | fedBN | fedrep | moon | fedproto | adcol | RUCR | fedheal | ours
- **--dataset** specify datasets, option: digit | office | PACS
- **--exp** experiment No.
- **--iters** iterations for communication
- **--batch** batch size of local training
- Of course, you can statically set other parameters through the parameter configuration file `./exps/option.py`
```bash
cd exps
# benchmark experiment in [digit | office | PACS] dataset
python federated_main.py --exp 1
```
You can run our ablation experiments by following the command.
```bash
cd exps
# ablation1 experiment in [digit | office | PACS] dataset : effect of loss combination
python federated_main.py --exp 2
# ablation2 experiment in [digit | office | PACS] dataset : whether to replace the local classifier
python federated_main.py --exp 3
```

### Test
You can use our trained weights file to calculate the accuracy of the test set. Our weights file is saved [here](https://drive.google.com/drive/folders/1bUJ-fwX2njgZnRDDjaDP0OXGTJc957Aw?usp=sharing). You need to put it in the ```./exps/weights``` folder.

The file structure after the weight file is downloaded is as follows，
```.
weights
├── 0                                                // Random Seed [0 | 1 | 2]
│   ├── PACS                                         // Dataset [digit | office | PACS]
│   │   ├── best_local_model_art_painting.pth        // The optimal model parameters for the client corresponding to the sub-dataset of a specific dataset
│   │   ├── best_local_model_cartoon.pth
│   │   ├── best_local_model_photo.pth
│   │   └── best_local_model_sketch.pth
│   ├── digit
│   │   ├── best_local_model_MNIST-M.pth
│   │   ├── best_local_model_MNIST.pth
│   │   ├── best_local_model_SVHN.pth
│   │   ├── best_local_model_SynthDigits.pth
│   │   └── best_local_model_USPS.pth
│   └── office
│       ├── best_local_model_amazon.pth
│       ├── best_local_model_caltech.pth
│       ├── best_local_model_dslr.pth
│       └── best_local_model_webcam.pth
```
Please use the following command to test our algorithm,
- **--dataset** specify datasets, option: digit | office | PACS
- **--batch** batch size of local training
```bash
cd exps
# test experiment in [digit | office | PACS] dataset
python test.py --dataset digit
```
## BibTeX
If you find AdvEncoder both interesting and helpful, please consider citing us in your research or publications:
```
@InProceedings{Zhang_2025_ICCV,
    author    = {Zhang, Yong and Liang, Feng and Yuan, Guanghu and Yang, Min and Li, Chengming and Hu, Xiping},
    title     = {FedPall: Prototype-based Adversarial and Collaborative Learning for Federated Learning with Feature Drift},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {1-10}
}
```
