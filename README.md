## Torchplate

This is another of an existing project of mine, that I thought would be useful for other people.

Working in the industry, you need to work on complex Deep Neural Networks (DNNs) projects and a script with all your training code just doesn't cut it. Many people use your code, so it needs to reusable, efficient, readable and scalable.
That is why I introduced (for myself, my colleagues and hopefully others if they find it useful) a template for Pytorch projects. I mostly work with Convolutional Neural Networks (CNNs) so this project is catered toward them.

Python 3.7

Torch 1.2 (last trained. I now use 1.4 but this repository has not been tested on 1.4 yet)

```
+-- config/
|   +-- train.toml
|   +-- evaluate.toml
+-- your_cnn_project
|   +-- dataset/
|   |   +-- dataset_1.py
|   +-- data_loader
|   |   +-- data_loader.py
|   +-- models
|   |   +-- model_1.py
|   |   +-- model_2.py
|   |   +-- pruned_model_1.py
|   +-- loss
|   |   +-- loss_functions.py
|   +-- trainer
|   |   +-- trainer.py
|   |   +-- trainer_multi_task.py
|   +-- evaluate
|   |   +-- evaluate_function.py
|   +-- transforms
|   |   +-- image_transforms.py
|   +-- utils
|   |   +-- utils.py
|   +-- train.py
+-- run.py
+-- evaluate.py
```

# Models

# Config

This project uses toml based configuration files for defining parameters for the training pipeline.

!! NOTE !! This project is not supposed to run as it is. It is a template to base your projects on but with as little changes as necassary.

# To Train
```
python run.py -c <path-to-config-file>
```

# For evaluation
```
python evaluate.py -c <path-to-config-file>
```
