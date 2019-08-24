# Lightning Module interface
[[Github Code](https://github.com/williamFalcon/pytorch-lightning/blob/master/pytorch_lightning/root_module/root_module.py)]

A lightning module is a strict superclass of nn.Module, it provides a standard interface for the trainer to interact with the model.

The easiest thing to do is copy the [minimal example](https://williamfalcon.github.io/pytorch-lightning/LightningModule/RequiredTrainerInterface/#minimal-example) below and modify accordingly. 

Otherwise, to Define a Lightning Module, implement the following methods:

**Required**:  

- [training_step](RequiredTrainerInterface.md#training_step)   
- [validation_step](RequiredTrainerInterface.md#validation_step)
- [validation_end](RequiredTrainerInterface.md#validation_end)
    
- [configure_optimizers](RequiredTrainerInterface.md#configure_optimizers)

- [tng_dataloader](RequiredTrainerInterface.md#tng_dataloader)
- [tng_dataloader](RequiredTrainerInterface.md#tng_dataloader)
- [test_dataloader](RequiredTrainerInterface.md#test_dataloader)

**Optional**:   

- [on_save_checkpoint](RequiredTrainerInterface.md#on_save_checkpoint)
- [on_load_checkpoint](RequiredTrainerInterface.md#on_load_checkpoint)
- [update_tng_log_metrics](RequiredTrainerInterface.md#update_tng_log_metrics)
- [add_model_specific_args](RequiredTrainerInterface.md#add_model_specific_args)

---
### Minimal example   
```python
import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

import pytorch_lightning as pl

class CoolModel(pl.LightningModule):

    def __init__(self):
        super(CoolModel, self).__init__()
        # not the best model...
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def my_loss(self, y_hat, y):
        return F.cross_entropy(y_hat, y)

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        return {'loss': self.my_loss(y_hat, y)}

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        return {'val_loss': self.my_loss(y_hat, y)}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'avg_val_loss': avg_loss}

    def configure_optimizers(self):
        return [torch.optim.Adam(self.parameters(), lr=0.02)]

    @pl.data_loader
    def tng_dataloader(self):
        return DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()), batch_size=32)

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()), batch_size=32)

    @pl.data_loader
    def test_dataloader(self):
        return DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()), batch_size=32)
```
---   
### How do these methods fit into the broader training?     
The LightningModule interface is on the right. Each method corresponds to a part of a research project. Lightning automates everything not in blue.    

<p align="center">
  <a href="https://github.com/williamFalcon/pytorch-lightning/blob/master/docs/source/_static/overview_flat.jpg">
    <img alt="" src="https://github.com/williamFalcon/pytorch-lightning/blob/master/docs/source/_static/overview_flat.jpg" height="900px">
  </a>
</p>   

---

### training_step

``` {.python}
def training_step(self, data_batch, batch_nb)
```

In this step you'd normally do the forward pass and calculate the loss for a batch. You can also do fancier things like multiple forward passes or something specific to your model.

**Params**    

| Param  | description  |
|---|---|
|  data_batch | The output of your dataloader. A tensor, tuple or list  |
|  batch_nb | Integer displaying which batch this is  |

**Return**   

Dictionary or OrderedDict   

| key  | value  | is required |
|---|---|---|
|  loss | tensor scalar  | Y |
|  prog | Dict for progress bar display. Must have only tensors  | N |


**Example**

``` {.python}
def training_step(self, data_batch, batch_nb):
    x, y, z = data_batch
    
    # implement your own
    out = self.forward(x)
    loss = self.loss(out, x)
    
    output = {
        'loss': loss, # required
        'prog': {'tng_loss': loss, 'batch_nb': batch_nb} # optional
    }
    
    # return a dict
    return output
```

---

### validation_step

``` {.python}
def validation_step(self, data_batch, batch_nb)
```

In this step you'd normally do the forward pass and calculate the loss for a batch. You can also do fancier things like multiple forward passes or something specific to your model.
This is most likely the same as your training_step. But unlike training step, the outputs from here will go to validation_end for collation.

**Params**   

| Param  | description  |
|---|---|
|  data_batch | The output of your dataloader. A tensor, tuple or list  |
|  batch_nb | Integer displaying which batch this is  |

**Return**   

| Return  | description  | optional |
|---|---|---|   
|  dict | Dict of OrderedDict with metrics to display in progress bar. All keys must be tensors. | Y |

**Example**

``` {.python}
def validation_step(self, data_batch, batch_nb):
    x, y, z = data_batch
    
    # implement your own
    out = self.forward(x)
    loss = self.loss(out, x)
    
    # calculate acc
    labels_hat = torch.argmax(out, dim=1)
    val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
    
    # all optional...
    # return whatever you need for the collation function validation_end
    output = OrderedDict({
        'val_loss': loss_val,
        'val_acc': torch.tensor(val_acc), # everything must be a tensor
    })
    
    # return an optional dict
    return output
```

--- 
### validation_end

``` {.python}
def validation_end(self, outputs)
```

Called at the end of the validation loop with the output of each validation_step.

**Params**    

| Param  | description  |
|---|---|
|  outputs | List of outputs you defined in validation_step |

**Return**   

| Return  | description  | optional |
|---|---|---|   
|  dict | Dict of OrderedDict with metrics to display in progress bar | Y |

**Example**

``` {.python}
def validation_end(self, outputs):
    """
    Called at the end of validation to aggregate outputs
    :param outputs: list of individual outputs of each validation step
    :return:
    """
    val_loss_mean = 0
    val_acc_mean = 0
    for output in outputs:
        val_loss_mean += output['val_loss']
        val_acc_mean += output['val_acc']

    val_loss_mean /= len(outputs)
    val_acc_mean /= len(outputs)
    tqdm_dic = {'val_loss': val_loss_mean.item(), 'val_acc': val_acc_mean.item()}
    return tqdm_dic
```

--- 
### configure_optimizers 

``` {.python}
def configure_optimizers(self)
```

Set up as many optimizers and (optionally) learning rate schedulers as you need. Normally you'd need one. But in the case of GANs or something more esoteric you might have multiple. 
Lightning will call .backward() and .step() on each one in every epoch.  If you use 16 bit precision it will also handle that.


##### Return
List or Tuple - List of optimizers with an optional second list of learning-rate schedulers

**Example**

``` {.python}
# most cases
def configure_optimizers(self):
    opt = Adam(self.parameters(), lr=0.01)
    return [opt]
    
# gan example, with scheduler for discriminator
def configure_optimizers(self):
    generator_opt = Adam(self.model_gen.parameters(), lr=0.01)
    disriminator_opt = Adam(self.model_disc.parameters(), lr=0.02)
    discriminator_sched = CosineAnnealing(discriminator_opt, T_max=10)
    return [generator_opt, disriminator_opt], [discriminator_sched]
```

--- 
### on_save_checkpoint 

``` {.python}
def on_save_checkpoint(self, checkpoint)
```
Called by lightning to checkpoint your model. Lightning saves the training state (current epoch, global_step, etc)
and also saves the model state_dict. If you want to save anything else, use this method to add your own
key-value pair.

##### Return
Nothing

**Example**

``` {.python}
def on_save_checkpoint(self, checkpoint):
    # 99% of use cases you don't need to implement this method 
    checkpoint['something_cool_i_want_to_save'] = my_cool_pickable_object
```

--- 
### on_load_checkpoint 

``` {.python}
def on_load_checkpoint(self, checkpoint)
```
Called by lightning to restore your model. Lighting auto-restores global step, epoch, etc...
It also restores the model state_dict.
If you saved something with **on_save_checkpoint** this is your chance to restore this.

##### Return
Nothing  

**Example**

``` {.python}
def on_load_checkpoint(self, checkpoint):
    # 99% of the time you don't need to implement this method
    self.something_cool_i_want_to_save = checkpoint['something_cool_i_want_to_save']
```

--- 
### tng_dataloader 

``` {.python}
@pl.data_loader
def tng_dataloader(self)
```
Called by lightning during training loop. Make sure to use the @pl.data_loader decorator, this ensures not calling this function until the data are needed.

##### Return
PyTorch DataLoader

**Example**

``` {.python}
@pl.data_loader
def tng_dataloader(self):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    dataset = MNIST(root='/path/to/mnist/', train=True, transform=transform, download=True)
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=self.hparams.batch_size,
        shuffle=True
    )
    return loader
```

--- 
### val_dataloader 

``` {.python}
@pl.data_loader
def tng_dataloader(self)
```
Called by lightning during validation loop. Make sure to use the @pl.data_loader decorator, this ensures not calling this function until the data are needed.

##### Return
PyTorch DataLoader

**Example**

``` {.python}
@pl.data_loader
def val_dataloader(self):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    dataset = MNIST(root='/path/to/mnist/', train=False, transform=transform, download=True)
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=self.hparams.batch_size,
        shuffle=True
    )
    
    return loader
```

--- 
### test_dataloader 

``` {.python}
@pl.data_loader
def test_dataloader(self)
```
Called by lightning during test loop. Make sure to use the @pl.data_loader decorator, this ensures not calling this function until the data are needed.

##### Return
PyTorch DataLoader

**Example**

``` {.python}
@pl.data_loader
def test_dataloader(self):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    dataset = MNIST(root='/path/to/mnist/', train=False, transform=transform, download=True)
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=self.hparams.batch_size,
        shuffle=True
    )
    
    return loader
```

--- 
### update_tng_log_metrics 

``` {.python}
def update_tng_log_metrics(self, logs)
```
Called by lightning right before it logs metrics for this batch.
This is a chance to ammend or add to the metrics about to be logged.

##### Return
Dict 

**Example**

``` {.python}
def update_tng_log_metrics(self, logs):
    # modify or add to logs
    return logs
```

--- 
### add_model_specific_args 

``` {.python}
@staticmethod
def add_model_specific_args(parent_parser, root_dir)
```
Lightning has a list of default argparse commands.
This method is your chance to add or modify commands specific to your model.
The [hyperparameter argument parser](https://williamfalcon.github.io/test-tube/hyperparameter_optimization/HyperOptArgumentParser/) is available anywhere in your model by calling self.hparams.

##### Return
An argument parser

**Example**

``` {.python}
@staticmethod
def add_model_specific_args(parent_parser, root_dir):
    parser = HyperOptArgumentParser(strategy=parent_parser.strategy, parents=[parent_parser])

    # param overwrites
    # parser.set_defaults(gradient_clip=5.0)

    # network params
    parser.opt_list('--drop_prob', default=0.2, options=[0.2, 0.5], type=float, tunable=False)
    parser.add_argument('--in_features', default=28*28)
    parser.add_argument('--out_features', default=10)
    parser.add_argument('--hidden_dim', default=50000) # use 500 for CPU, 50000 for GPU to see speed difference

    # data
    parser.add_argument('--data_root', default=os.path.join(root_dir, 'mnist'), type=str)

    # training params (opt)
    parser.opt_list('--learning_rate', default=0.001, type=float, options=[0.0001, 0.0005, 0.001, 0.005],
                    tunable=False)
    parser.opt_list('--batch_size', default=256, type=int, options=[32, 64, 128, 256], tunable=False)
    parser.opt_list('--optimizer_name', default='adam', type=str, options=['adam'], tunable=False)
    return parser
```
