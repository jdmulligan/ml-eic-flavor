# Running the code

## Set up software environment

Initialize a python environment with the packages listed in Pipfile. For example:
```
cd ml-eic-flavor
pipenv install                                # (this step is only needed the first time)
pipenv shell
```

The setup was designed to run on hiccup -- some modifications to the python setup might be needed if running elsewhere. 
Please note that one should not run anything other than very quick small scripts on hiccup0 -- for more substantial runs like with the full ML training statistics you should `ssh hiccup1` and let James know in order to make sure we don't overload a node.

## q-g jets from Kyle

The jet samples are located in the `training_data` folder.

The analysis workflow is as follows:

1. Edit the config file `config/qg/yaml`. In particular:
    - Set `n_train, n_val, n_test` to the desired size of the training sample
    - Set `models` to contain the ML architectures you want to include
    - Edit the config blocks for each model as desired
   
2. Fit model and make plots:
   ```
   cd ml-eic-flavor
   python analyze_flavor.py -c <config> -o <output_dir>
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
