Lightning makes multi-gpu training and 16 bit training trivial.

*Note:*   
None of the flags below require changing anything about your lightningModel definition. 

---
#### Choosing a backend  
Lightning supports two backends. DataParallel and DistributedDataParallel. Both can be used for single-node multi-GPU training.
For multi-node training you must use DistributedDataParallel.   

You can toggle between each mode by setting this flag.
``` {.python}
# DEFAULT uses DataParallel
trainer = Trainer(distributed_backend='dp')

# change to distributed data parallel
trainer = Trainer(distributed_backend='ddp')
```

If you request multiple nodes, the back-end will auto-switch to ddp.
We recommend you use DistributedDataparallel even for single-node multi-GPU training. It is MUCH faster than DP but *may*
have configuration issues depending on your cluster.

For a deeper understanding of what lightning is doing, feel free to read [this guide](https://medium.com/@_willfalcon/9-tips-for-training-lightning-fast-neural-networks-in-pytorch-8e63a502f565).   

---
#### Distributed and 16-bit precision.    
Due to an issue with apex and DistributedDataParallel (PyTorch and NVIDIA issue), Lightning does
not allow 16-bit and DP training. We tried to get this to work, but it's an issue on their end.   

Below are the possible configurations we support.    

| 1 GPU  | 1+ GPUs  | DP  | DDP  | 16-bit  | command |
|---|---|---|---|---|---|
| Y  |   |   |   |  | ```Trainer(gpus=[0])``` |
| Y  |   |   |   | Y | ```Trainer(gpus=[0], use_amp=True)``` |
|   | Y | Y |   |   | ```Trainer(gpus=[0, ...])``` |
|   | Y |  | Y  |  | ```Trainer(gpus=[0, ...], distributed_backend='ddp')``` |
|   | Y |  | Y  | Y | ```Trainer(gpus=[0, ...], distributed_backend='ddp', use_amp=True)``` |


---
#### CUDA flags   
CUDA flags make certain GPUs visible to your script. 
Lightning sets these for you automatically, there's NO NEED to do this yourself.
```python
# lightning will set according to what you give the trainer
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```

---
#### 16-bit mixed precision
16 bit precision can cut your memory footprint by half. If using volta architecture GPUs it can give a dramatic training speed-up as well.    
First, install apex (if install fails, look [here](https://github.com/NVIDIA/apex)):
```bash
$ git clone https://github.com/NVIDIA/apex
$ cd apex
$ pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

then set this use_amp to True.
``` {.python}
# DEFAULT
trainer = Trainer(amp_level='O2', use_amp=False)
```

---
#### Single-gpu
Make sure you're on a GPU machine. 
```python
# DEFAULT
trainer = Trainer(gpus=[0])
```

---
#### multi-gpu 
Make sure you're on a GPU machine. You can set as many GPUs as you want.
In this setting, the model will run on all 8 GPUs at once using DataParallel under the hood.
```python
# to use DataParallel (default)
trainer = Trainer(gpus=[0,1,2,3,4,5,6,7], distributed_backend='dp')

# RECOMMENDED use DistributedDataParallel
trainer = Trainer(gpus=[0,1,2,3,4,5,6,7], distributed_backend='ddp')
```

---
#### Multi-node
Multi-node training is easily done by specifying these flags.
```python
# train on 12*8 GPUs
trainer = Trainer(gpus=[0,1,2,3,4,5,6,7], nb_gpu_nodes=12)
```

In addition, make sure to set up your SLURM job correctly via the [SlurmClusterObject](https://williamfalcon.github.io/test-tube/hpc/SlurmCluster/). In particular, specify the number of tasks per node correctly.

```python
cluster = SlurmCluster(
    hyperparam_optimizer=test_tube.HyperOptArgumentParser(),
    log_path='/some/path/to/save',
)

# OPTIONAL FLAGS WHICH MAY BE CLUSTER DEPENDENT
# which interface your nodes use for communication
cluster.add_command('export NCCL_SOCKET_IFNAME=^docker0,lo')

# see output of the NCCL connection process
# NCCL is how the nodes talk to each other
cluster.add_command('export NCCL_DEBUG=INFO')

# setting a master port here is a good idea.
cluster.add_command('export MASTER_PORT=%r' % PORT)

# good to load the latest NCCL version
cluster.load_modules(['NCCL/2.4.7-1-cuda.10.0'])

# configure cluster
cluster.per_experiment_nb_nodes = 12 
cluster.per_experiment_nb_gpus = 8

cluster.add_slurm_cmd(cmd='ntasks-per-node', value=8, comment='1 task per gpu')
```

Finally, make sure to add a distributed sampler to your dataset. The distributed sampler copies a 
portion of your dataset onto each GPU. (World_size = gpus_per_node * nb_nodes).   

```python
# ie: this:
dataset = myDataset()
dataloader = Dataloader(dataset)

# becomes:
dataset = myDataset()
dist_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
dataloader = Dataloader(dataset, sampler=dist_sampler)
```

---
#### Self-balancing architecture
Here lightning distributes parts of your module across available GPUs to optimize for speed and memory.   

COMING SOON.
