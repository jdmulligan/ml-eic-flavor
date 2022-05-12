# Running the code

## Setup software environment – on hiccup cluster

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
Please do not run anything bust the lightest tests on the login node. If you are finding that you have to wait a long time, let us know and we can take a node out of the slurm queue and logon to it directly.

### Initialize environment
  
Now we need to initialize the environment: load heppy, set the python version, and create a virtual environment for python packages.
Since various ML packages require higher python versions than installed system-wide, we have set up an initialization script to take care of this. 
The first time you set up, you can do:
```
cd ml-eic-flavor
./init.sh --install
```
  
On subsequent times, you don't need to pass the `install` flag:
```
cd ml-eic-flavor
./init.sh
```

Now we are ready to run our scripts.

## q-g jets from Kyle

The jet samples are located in the `training_data` folder.

The analysis workflow is as follows:

1. Edit the config file `config/qg.yaml`. In particular:
    - Set `n_train, n_val, n_test` to the desired size of the training sample
    - Set `models` to contain the ML architectures you want to include
    - Edit the config blocks for each model as desired
   
2. Fit model and make plots:
   ```
   cd ml-eic-flavor
   python analysis/analyze_flavor.py -c config/qg.yaml -o <output_dir>
   ```
   The `-o` path is the location that the output plots will be written to. 

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
