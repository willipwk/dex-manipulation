# CMPT720 Project: Dexterous Manipulation

## Installation
1. Follow [dex-retargeting](https://github.com/dexsuite/dex-retargeting/tree/main) and [retarget from hand object pose dataset](https://github.com/dexsuite/dex-retargeting/blob/main/example/position_retargeting/README.md) to clone the repo, install required dependencies, prepare the DexYCB dataset.
2. Download files in this repo and put them in `dex-retargeting/examples/position_retargeting`.
3. Install nimblephysics, Genesis, and rsl-rl-lib.
    ```bash
    pip install nimblephysics genesis-world==0.4.4 rsl-rl-lib==2.2.4
    ```

## Retargeting
You can follow [retarget from hand object pose dataset](https://github.com/dexsuite/dex-retargeting/blob/main/example/position_retargeting/README.md) to retarget hand-object trajectories in the DexYCB dataset to robot hand poses. 

## Differentiable Simulation
To run differentiable simulation experiment, first run hand tracking optimization
```bash
python test_nimble_single.py
```
It will save the controls in a npy file. Then, run hand-object trajectory optimization
```bash
python test_nimble_seq.py
```
You may need to update paths in the python script.

## MPPI
To run MPPI experiment, run
```bash
python mppi_hand_force.py --dexycb_dir YOUR_DEXYCB_DIR
```

## RL
to run RL experiment, run
```bash
python rl_hand_train.py --dexycb_dir YOUR_DEXYCB_DIR
```