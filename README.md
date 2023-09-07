# Mesh VAE for cardiac anatomy modeling

This is an implementation of the variational mesh autoencoder (Mesh VAE) for cardiac anatomy modeling as described in   
   
**Interpretable cardiac anatomy modeling using variational mesh autoencoders.**
Marcel Beetz, Jorge Corral Acero, Abhirup Banerjee, Ingo Eitel, Ernesto Zacur, Torben Lange, Thomas Stiermaier, Ruben Evertz, Sören J. Backhaus, Holger Thiele, Alfonso Bueno-Orovio, Pablo Lamata, Andreas Schuster, and Vicente Grau.
*Frontiers in Cardiovascular Medicine.*
[[Paper]](https://www.frontiersin.org/articles/10.3389/fcvm.2022.983868/full)

## Requirements
The code was tested with Python 3.6.9 and Pytorch 1.3.0. 
Before running the code, please ensure that the requirements are satisfied by following these steps:
1. Install the mesh processing libraries from [MPI-IS/mesh](https://github.com/MPI-IS/mesh)
2. Install the required python packages by running:
```bash
pip install -r requirements.txt
```

## Data
The datasets used in this work are not publicly available. Generated virtual data can be made available upon reasonable request. A sample template mesh is provided as a reference (`./data/sample_template_mesh.ply`). All meshes in the dataset should have the same vertex connectivity as the template mesh.

## Training
To train the network, follow these steps:
1. Adjust the following parameters in the `settings.cfg` file or pass them as command line arguments in step 3:
    - `data_dir` - set path to where dataset is located
    - `checkpoint_dir` - set path to where checkpoints will be saved
    - `output_dir` - set path to where ground truth and predicted meshes will be stored if `visualize` is set to `True` 
    - `eval` - set to `False` for training
2. Adjust the remaining parameters in the `settings.cfg` file as required
3. Start training by running: 
```bash
python main.py
```

## Testing
To evaluate a trained network on the test dataset, follow these steps:
1. In config file `settings.cfg`: Set `eval` to `True`
2. In config file `settings.cfg`: Set `checkpoint_file` to path where model weights are stored
3. Start testing by running:
```bash
python main.py
```

## Citation
If you find this work useful, please cite:
```
@ARTICLE{10.3389/fcvm.2022.983868,
    AUTHOR={Beetz, Marcel and Corral Acero, Jorge and Banerjee, Abhirup and Eitel, Ingo and Zacur, Ernesto and Lange, Torben and Stiermaier, Thomas and Evertz, Ruben and Backhaus, Sören J. and Thiele, Holger and Bueno-Orovio, Alfonso and Lamata, Pablo and Schuster, Andreas and Grau, Vicente},    
    TITLE={Interpretable cardiac anatomy modeling using variational mesh autoencoders},      
    JOURNAL={Frontiers in Cardiovascular Medicine},      
    VOLUME={9},           
    YEAR={2022},        
    URL={https://www.frontiersin.org/articles/10.3389/fcvm.2022.983868},       
    DOI={10.3389/fcvm.2022.983868},      
    ISSN={2297-055X},   
}
```

## Acknowledgements
Parts of this code are based on software from other repositories. Please see the [ACKNOWLEDGEMENTS](ACKNOWLEDGEMENTS.txt) file for more details.

## License
[MIT](LICENSE.txt)
