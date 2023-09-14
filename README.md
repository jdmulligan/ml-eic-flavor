# Jet classification at the Electron-Ion Collider

This repository contains analysis code for [arXiv:2210.06450](https://inspirehep.net/literature/2164495), performing classification of high-energy jets at the Electron-Ion Collider using deep sets / point clouds.

The training data sets used in this work can be found [here](10.5281/zenodo.7538810).

# Running the code

The analysis workflow is as follows:

1. Setup the software environment. A couple example instructions are below; adapt them as appropriate.

2. Create a config file. There are examples for all studies in the `config` directory, e.g. e.g. `config/ud_s.yaml`. In particular, you will want to:
    - Download the data and set the paths of the training files under `input_files`
    - Set the event type and training classes
    - Set `models` to contain the ML architectures you want to include, and edit the config blocks for each model as desired
    - Edit other settings as desired. See the examples for further details.
   
3. Fit model and make plots:
   ```
   cd ml-eic-flavor
   python analysis/analyze_flavor.py -c config/my_config.yaml -o <output_dir>
   ```
   The `-o` path is the location that the output plots will be written to. 

## Setup software environment – on hiccup cluster
<details>
  <summary>Click for details</summary>
<br/> 
  
### Logon and allocate a node
  
Logon to hiccup:
```
ssh <user>@hic.lbl.gov
```

First, request an interactive node from the slurm batch system:
   ```
   srun -N 1 -n 20 -t 2:00:00 -p quick --pty bash
   ``` 
   which requests 1 full node (20 cores) for 2 hours in the `quick` queue. You can choose the time and queue: you can use the `quick` partition for up to a 2 hour session, `std` for a 24 hour session, or `long` for a 72 hour session – but you will wait longer for the longer queues). 
Depending how busy the squeue is, you may get the node instantly, or you may have to wait awhile.
When you’re done with your session, just type `exit`.
Please do not run anything but the lightest tests on the login node. If you are finding that you have to wait a long time, let us know and we can take a node out of the slurm queue and logon to it directly.

### Initialize environment
  
Now we need to initialize the environment: load heppy, set the python version, and create a virtual environment for python packages.
Since various ML packages require higher python versions than installed system-wide, we have set up an initialization script to take care of this. 
The first time you set up, you can do:
```
cd ml-eic-flavor
./init.sh --install
```
If you encounter an error that refers to python 3.6, make sure you do not have any modules or environments loaded (e.g. in your .bashrc). If you get an error related to pipenv, you can also try `pip install --user pipenv`.
  
On subsequent times, you don't need to pass the `install` flag:
```
cd ml-eic-flavor
./init.sh
```

Now we are ready to run our scripts.

   
</details>

## Setup software environment – on perlmutter
<details>
  <summary>Click for details</summary>
<br/> 
  
### Logon and allocate a node
  
Logon to perlmutter:
```
ssh <user>@perlmutter-p1.nersc.gov
```

First, request an [interactive node](https://docs.nersc.gov/jobs/interactive/) from the slurm batch system:
   ```
   salloc --nodes 1 --qos interactive --time 04:00:00 --constraint gpu --gpus 4 --account=alice_g
   ``` 
   which requests 4 GPUs on a node in the alice allocation (perlmutter usage is not charged, so use as much as you want). 
When you’re done with your session, just type `exit`.

### Initialize environment
  
We will only run the ML part of the pipeline on perlmutter. For now, you should copy your output file of generated jets/events:
```
scp -r /rstorage/ml-eic-flavor/<output_file> <user>@perlmutter-p1.nersc.gov:/pscratch/sd/j/<user>/
```

Now we need to initialize the environment:
```
cd ml-eic-flavor
source init_perlmutter.sh
```

Now we are ready to run our scripts.

</details>

# Pull requests

To contribute code to the repository, you should make a pull request from your fork of the repo.

For example, working with a single branch:

1. First, create your fork of `jdmulligan/ml-eic-flavor` by clicking the "Fork" tab in the upper right of the `jdmulligan/ml-eic-flavor` webpage.

2. Next, go to your local ml-eic-flavor repository (doing `git clone` from the original `jdmulligan/ml-eic-flavor`). We want to add a "remote" to your fork. To see your current remotes, you can type `git remote -v`. 

- To add a remote to your fork, type `git remote add myFork git@github.com:<github-username>/ml-eic-flavor.git` (or using https, if you use that)

3. Suppose you now have a commit on your local repository that you want to propose to add to the main `ml-eic-flavor` repository.

- First, we should update our local code to the latest changes in `jdmulligan/ml-eic-flavor` and put our commit on top of that: `git pull --rebase`
- Next, commit to your fork: `git push myFork master`
- Then, go to your fork and create a pull request.
