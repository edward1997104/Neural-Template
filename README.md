# Neural-Template
Neural Template: Topology-aware Reconstruction and Disentangled Generation of 3D Meshes (CVPR 2022)

![gallery](figures/gallery.png)

# Environments
You can create an conda environment for this project using the following commands:
```angular2html
conda create -n NeuralTemplate -r requirement.ymal
```

# Dataset
For the dataset, we use dataset provided by [IM-NET](https://github.com/czq142857/IM-NET-pytorch) for training and evaluation. We provide another zip file ([link](https://drive.google.com/file/d/1--C2xUp0yao_nHDNvEpL3a1ZpTVC139J/view?usp=sharing)) which should be unzipped in ```data``` before any training or inference.

# Training
There are all together three commands for three training phases (Continuous training, Discrete training, Image encoder training).

For the continuous phase, you can train the model by using the following command:
```angular2html
python train/implicit_trainer.py --resume_path ./configs/config_continuous.py
```

For the discrete phase, you should first specify the ```network_resume_path``` variable in the ```configs/config_discrete.py``` as the model path in the previous phase. Then, you can run the following command:
```angular2html
python train/implicit_trainer.py --resume_path ./configs/config_discrete.py
```

Lastly for the Image encoder training, you should first specify the ```auto_encoder_config_path```  and ```auto_encoder_resume_path``` in ```config_image.py``` similiarly, and run the following command to start training:
```angular2html
python train/image_trainer.py --resume_path ./configs/config_image.py
```

# Testing
We provide the pre-trained models used in the paper for reproducing the results. You can unzip the file ([link](https://drive.google.com/file/d/1--C2xUp0yao_nHDNvEpL3a1ZpTVC139J/view?usp=sharing)) in the ```pretrain``` folder.

For the evaluation of the autoencoder, you can use this command:
```angular2html
python utils/mesh_visualization.py
```

For the evaluation of the single view reconstruction task, you can use this command:
```angular2html
python utils/image_mesh_visualization.py
```

If you need test your own trained model, you change the ```testing_folder``` variable in these two python files.


### Citation
If you find our work useful in your research, please consider citing:
```
@inproceedings{hui2022template,
    title = {Neural Template: Topology-aware Reconstruction and Disentangled Generation of 3D Meshes},
    author = {Ka-Hei Hui* and Ruihui Li and Jingyu Wu and Chi-Wing Fu(* joint first authors)},
    booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2022}
}
```


