### Template model definition
In 99% of cases you want to just copy [this template](https://github.com/williamFalcon/pytorch-lightning/blob/master/examples/new_project_templates/lightning_module_template.py) to start a new lightningModule and change the core of what your model is actually trying to do.

```bash
# get a copy of the module template
wget https://github.com/williamFalcon/pytorch-lightning/blob/master/examples/new_project_templates/lightning_module_template.py
```

---    
### Trainer Example 

** \_\_main__ function**    

Normally, we want to let the \_\_main__ function start the training.
Inside the main we parse training arguments with whatever hyperparameters we want. Your LightningModule will have a 
chance to add hyperparameters.   

```{.python}
from test_tube import HyperOptArgumentParser

if __name__ == '__main__':

    # use default args given by lightning
    root_dir = os.path.split(os.path.dirname(sys.modules['__main__'].__file__))[0]
    parent_parser = HyperOptArgumentParser(strategy='random_search', add_help=False)
    add_default_args(parent_parser, root_dir)

    # allow model to overwrite or extend args
    parser = ExampleModel.add_model_specific_args(parent_parser)
    hyperparams = parser.parse_args()

    # train model
    main(hyperparams)
```
**Main Function**      

The main function is your entry into the program. This is where you init your model, checkpoint directory, and launch the training.
The main function should have 3 arguments:   
- hparams: a configuration of hyperparameters.    
- slurm_manager: Slurm cluster manager object (can be None)
- dict: for you to return any values you want (useful in meta-learning, otherwise set to _)    

```{}
def main(hparams, cluster, results_dict):
    """
    Main training routine specific for this project
    :param hparams:
    :return:
    """
    # init experiment
    log_dir = os.path.dirname(os.path.realpath(__file__))
    exp = Experiment(
        name='test_tube_exp',
        debug=True,
        save_dir=log_dir,
        version=0,
        autosave=False,
        description='test demo'
    )
    
    # set the hparams for the experiment
    exp.argparse(hparams)
    exp.save()

    # build model
    model = MyLightningModule(hparams)

    # callbacks
    early_stop = EarlyStopping(
        monitor=hparams.early_stop_metric,
        patience=hparams.early_stop_patience,
        verbose=True,
        mode=hparams.early_stop_mode
    )

    model_save_path = '{}/{}/{}'.format(hparams.model_save_path, exp.name, exp.version)
    checkpoint = ModelCheckpoint(
        filepath=model_save_path,
        save_function=None,
        save_best_only=True,
        verbose=True,
        monitor=hparams.model_save_monitor_value,
        mode=hparams.model_save_monitor_mode
    )

    # configure trainer
    trainer = Trainer(
        experiment=exp,
        cluster=cluster,
        checkpoint_callback=checkpoint,
        early_stop_callback=early_stop,
    )

    # train model
    trainer.fit(model)
```




The __main__ function will start training on your **main** function. If you use the HyperParameterOptimizer
in hyper parameter optimization mode, this main function will get one set of hyperparameters. If you use it as a simple
argument parser you get the default arguments in the argument parser.

So, calling main(hyperparams) runs the model with the default argparse arguments.       
```{.python}
main(hyperparams)
```

---
#### CPU hyperparameter search      

```{.python}
# run a grid search over 20 hyperparameter combinations.
hyperparams.optimize_parallel_cpu(
    main_local,
    nb_trials=20,
    nb_workers=1
)
```

---
#### Hyperparameter search on a single or multiple GPUs       
```{.python}
# run a grid search over 20 hyperparameter combinations.
hyperparams.optimize_parallel_gpu(
    main_local,
    nb_trials=20,
    nb_workers=1,
    gpus=[0,1,2,3]
)
```

---
#### Hyperparameter search on a SLURM HPC cluster   
```{.python}    
def optimize_on_cluster(hyperparams):
    # enable cluster training
    cluster = SlurmCluster(
        hyperparam_optimizer=hyperparams,
        log_path=hyperparams.tt_save_path,
        test_tube_exp_name=hyperparams.tt_name
    )

    # email for cluster coms
    cluster.notify_job_status(email='add_email_here', on_done=True, on_fail=True)

    # configure cluster
    cluster.per_experiment_nb_gpus = hyperparams.per_experiment_nb_gpus
    cluster.job_time = '48:00:00'
    cluster.gpu_type = '1080ti'
    cluster.memory_mb_per_node = 48000

    # any modules for code to run in env
    cluster.add_command('source activate pytorch_lightning')

    # name of exp
    job_display_name = hyperparams.tt_name.split('_')[0]
    job_display_name = job_display_name[0:3]

    # run hopt
    print('submitting jobs...')
    cluster.optimize_parallel_cluster_gpu(
        main,
        nb_trials=hyperparams.nb_hopt_trials,
        job_name=job_display_name
    )

# run cluster hyperparameter search    
optimize_on_cluster(hyperparams)
```
