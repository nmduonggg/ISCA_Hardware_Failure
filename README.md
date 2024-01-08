# Understanding and Mitigating Hardware Failures in Deep Learning Training Accelerator Systems

## Step 1. Download files from this [link](https://doi.org/10.5281/zenodo.7952090)

Then move to server with:

```sh
rsync -avz /path/to/local/folder/ username@remote:/path/on/remote/server
```

Extract with:

```sh
tar -xf /path/to/yourfile.tar
```

## Step 2: Create conda env

```sh
conda create -n ISCA python=3.8 tensorflow=2.6.0 numpy=1.19.5  gdown 4.6.4
```

## Step 3: Double check

```sh
python3 -c "import numpy;print(numpy.__version__);import tensorflow as tf;print(tf.__version__); import gdown; print(gdown.__version__)"
```

## Step 4: Download checkpoint from Google Drive

```sh
cd fault_injection
gdown --folder https://drive.google.com/drive/folders/1HVRFWY7NI5xr5qzR8yNeSKCRVnJNnqFf?usp=sharing
```

## Step 5: Replace TPU with GPUs

(Optional) Access .py file. Recommended access directly in editor
```
vim fault_injection/reproduce_injection.py
```

Replace all following lines:

```sh
from local_tpu_resolver import LocalTPUClusterResolver
```

To:

```py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"  # Use GPUs with indices 1 and 2 or "0" with gpu:0
```

and: 

```py
tpu_name = os.getenv('TPU_NAME')
resolver = LocalTPUClusterResolver()
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.TPUStrategy(resolver)
per_replica_batch_size = config.BATCH_SIZE // strategy.num_replicas_in_sync
print("Finish TPU strategy setting!")
```

To: 

```py
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    print("Available GPUs:", gpus)
else:
    print("No GPUs found. Make sure TensorFlow is configured to use GPUs.")

strategy = tf.distribute.MirroredStrategy()
```

## Step 6: run example

- Example 1 (takes approximately 5 minutes).

```py
cd fault_injection
python3 reproduce_injections.py --file injections/inj_immediate_infs_nans.csv
```

- Example 2 (takes approximately 10-15 minutes).

```py
cd fault_injection
python3 reproduce_injections.py --file injections/inj_masked.csv
```

Example 3 (takes approximately 10-15 minutes).

```py
cd fault_injection
python3 reproduce_injections.py --file injections/inj_slow_degrade.csv
```

## Step 7: Evaluation

##### Before running any script, change TPU code to GPUs code similar to Step 5

```sh
cd technique/detection
python3 detection.py
python3 detection.py -- check
python3 calc_overhead.py
```

```sh
cd technique / replay
python3 replay . py
python3 replay . py -- rerun
python3 calc_overhead . py
```
