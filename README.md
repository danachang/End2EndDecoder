# End2EndDecoder
This is for end to end decoder for endoscopic imaging with out-of-focus noise. 
Note that this repository is not the finalized official version (as the paper is in preparation). 
Most of my research projects were stored in GitHub Enterprise (@github.mit.edu), so I moved this repository here. 



## Prerequisites

- Python 3.6
- [Tensorflow 1.15.0](https://github.com/tensorflow/tensorflow/)
- [NumPy](http://www.numpy.org/)
- [colorlog](https://pypi.org/project/colorlog/)
- [imageio](https://imageio.github.io/)

## Usage

### Organize datasets

- Create a folder under `./data/` for each mouse.
- Create three sub-folders: `train/`, `val/`, and `test/`.
- Create `.txt` file (like `./data/mouse1/train/label1.txt`) for each sub-folder. Each row records: image name, location x, location y, orientation, velocity.

### Train models with downloaded datasets:
```bash
python trainer.py --dataset_path [default: data/mouse1] --batch_size 36 --num_d_conv 6 --num_d_fc 3 --loss_type l1
```
- The configuration can be found in `config.py`.

### Interpret TensorBoard
Launch TensorBoard and go to the specified port, you can see different the loss in the **scalars** tab.
