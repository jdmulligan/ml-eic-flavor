#!/usr/bin/env python3

"""
Class to read jets from file, train ML models, and plot
"""

import os
import sys
import argparse
import yaml
import pickle
import subprocess
from numba import jit, prange
import functools
import shutil
from collections import defaultdict

# Data analysis and plotting
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_context('paper', rc={'font.size':18,'axes.titlesize':18,'axes.labelsize':18})

# Energy flow package
import energyflow
import energyflow.archs

# sklearn
import sklearn
import sklearn.linear_model
import sklearn.ensemble
import sklearn.model_selection
import sklearn.pipeline

# Tensorflow and Keras
import tensorflow as tf
from tensorflow import keras
import keras_tuner

# Base class
sys.path.append('.')
from base import common_base

#--------------------------------------------------------------- 
# Create a copy of four-vectors with a min-pt cut
#---------------------------------------------------------------        
@jit(nopython=True) 
def filter_four_vectors(X_particles, min_pt=0.):

    n_jets = X_particles.shape[0]
    n_particles = 800
    
    for i in prange(n_jets):
        jet = X_particles[i]

        new_jet_index = 0
        new_jet = np.zeros(jet.shape)

        for j in prange(n_particles):
            if jet[j][0] > min_pt:
                 new_jet[new_jet_index] = jet[j]
                 new_jet_index += 1
        
        X_particles[i] = new_jet.copy()

    return X_particles

################################################################
class AnalyzePPAA(common_base.CommonBase):

    #---------------------------------------------------------------
    # Constructor
    #---------------------------------------------------------------
    def __init__(self, config_file='', output_dir='', **kwargs):
        super(common_base.CommonBase, self).__init__(**kwargs)
        
        self.config_file = config_file
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Initialize config file
        self.initialize_config()
            
        self.filename_base = 'training_data/jets_pT{}.txt'

        # Remove keras-tuner folder, if it exists
        if os.path.exists('keras_tuner'):
            shutil.rmtree('keras_tuner')

        print(self)
        print()
        
    #---------------------------------------------------------------
    # Initialize config file into class members
    #---------------------------------------------------------------
    def initialize_config(self):
    
        # Read config file
        with open(self.config_file, 'r') as stream:
          config = yaml.safe_load(stream)
          
        self.jet_pt_min_list= config['jet_pt_min_list']
        self.min_particle_pt = config['min_particle_pt']
        self.jetR = 0.4
        self.kappa = config['kappa']

        self.q_label = config['q_label']
        self.g_label = config['g_label']

        self.n_train = config['n_train']
        self.n_val = config['n_val']
        self.n_test = config['n_test']
        self.n_total = self.n_train + self.n_val + self.n_test
        self.test_frac = 1. * self.n_test / self.n_total
        self.val_frac = 1. * self.n_val / (self.n_train + self.n_val)
        
        self.dmax = config['dmax']
        self.efp_measure = config['efp_measure']
        self.efp_beta = config['efp_beta']

        self.random_state = None  # seed for shuffling data (set to an int to have reproducible results)

        # Initialize model-specific settings
        self.config = config
        self.models = config['models']
        self.model_settings = {}
        for model in self.models:
            self.model_settings[model] = {}
            
            if 'dnn' in model:
                self.model_settings[model]['loss'] = config[model]['loss']
                self.model_settings[model]['learning_rate'] = config[model]['learning_rate']
                self.model_settings[model]['epochs'] = config[model]['epochs']
                self.model_settings[model]['batch_size'] = config[model]['batch_size']
                self.model_settings[model]['metrics'] = config[model]['metrics']
            
            if 'linear' in model:
                self.model_settings[model]['sgd_loss'] = config[model]['sgd_loss']
                self.model_settings[model]['sgd_penalty'] = config[model]['sgd_penalty']
                self.model_settings[model]['sgd_alpha'] = [float(x) for x in config[model]['sgd_alpha']]
                self.model_settings[model]['sgd_max_iter'] = config[model]['sgd_max_iter']
                self.model_settings[model]['sgd_tol'] = [float(x) for x in config[model]['sgd_tol']]
                self.model_settings[model]['sgd_learning_rate'] = config[model]['sgd_learning_rate']
                self.model_settings[model]['sgd_early_stopping'] = config[model]['sgd_early_stopping']
                self.model_settings[model]['n_iter'] = config[model]['n_iter']
                self.model_settings[model]['cv'] = config[model]['cv']
                self.model_settings[model]['lda_tol'] = [float(x) for x in config[model]['lda_tol']]

            if 'lasso' in model:
                self.model_settings[model]['alpha'] = config[model]['alpha']
                self.model_settings[model]['max_iter'] = config[model]['max_iter']
                self.model_settings[model]['tol'] = float(config[model]['tol'])
                self.model_settings[model]['n_iter'] = config[model]['n_iter']
                self.model_settings[model]['cv'] = config[model]['cv']
                if 'nsub' in model:
                    self.K_lasso = config[model]['K_lasso']
                if 'efp' in model:
                    self.d_lasso = config[model]['d_lasso']

            if model == 'pfn':
                self.model_settings[model]['Phi_sizes'] = tuple(config[model]['Phi_sizes'])
                self.model_settings[model]['F_sizes'] = tuple(config[model]['F_sizes'])
                self.model_settings[model]['epochs'] = config[model]['epochs']
                self.model_settings[model]['batch_size'] = config[model]['batch_size']
                self.model_settings[model]['use_pids'] = config[model]['use_pids']
                
            if model == 'efn':
                self.model_settings[model]['Phi_sizes'] = tuple(config[model]['Phi_sizes'])
                self.model_settings[model]['F_sizes'] = tuple(config[model]['F_sizes'])
                self.model_settings[model]['epochs'] = config[model]['epochs']
                self.model_settings[model]['batch_size'] = config[model]['batch_size']
                self.model_settings[model]['learning_rate'] = config[model]['learning_rate']

    #---------------------------------------------------------------
    # Main processing function
    #---------------------------------------------------------------
    def analyze_pp_aa(self):

        # Loop through combinations of event type, jetR, and R_max
        self.AUC = {}
        for jet_pt_min in self.jet_pt_min_list:

            # Skip if no models are selected
            if not self.models:
                continue
        
            # Clear variables
            self.y = None
            self.X_particles = None

            # Create output dir
            self.output_dir_i = os.path.join(self.output_dir, f'pt{jet_pt_min}')
            if not os.path.exists(self.output_dir_i):
                os.makedirs(self.output_dir_i)

            # Read input file into dataframe -- the files store the particle info as: (pt, eta, phi, pid)
            # Then transform these into a 3D numpy array (jets, particles, particle info)
            # The format of the particle info in X_particles will be: (pt, eta, phi, m, pid, charge)
            jet_df = pd.read_csv(self.filename_base.format(jet_pt_min), delimiter="\s+")                                                                                                                                                                                                               
            X_particles_total = self.create_jet_array(jet_df)                                                                                                                            
            self.y_total = jet_df[jet_df.ct==1]['qg'].to_numpy()

            # Determine total number of jets
            total_jets = int(self.y_total.size)
            total_jets_q = int(np.sum(self.y_total))
            total_jets_g = total_jets - total_jets_q
            print(f'Total number of jets available: {total_jets_q} (q), {total_jets_g} (g)')

            # If there is an imbalance, remove excess jets
            if total_jets_q > total_jets_g:
                indices_to_remove = np.where( np.isclose(self.y_total,1) )[0][total_jets_g:]
            elif total_jets_q < total_jets_g:
                indices_to_remove = np.where( np.isclose(self.y_total,0) )[0][total_jets_q:]
            y_balanced = np.delete(self.y_total, indices_to_remove)
            X_particles_balanced = np.delete(X_particles_total, indices_to_remove, axis=0)
            total_jets = int(y_balanced.size)
            total_jets_q = int(np.sum(y_balanced))
            total_jets_g = total_jets - total_jets_q
            print(f'Total number of jets available after balancing: {total_jets_q} (q), {total_jets_g} (g)')

            # Shuffle dataset 
            idx = np.random.permutation(len(y_balanced))
            if y_balanced.shape[0] == idx.shape[0]:
                y_shuffled = y_balanced[idx]
                X_particles_shuffled = X_particles_balanced[idx]
            else:
                print(f'MISMATCH of shape: {y_shuffled.shape} vs. {idx.shape}')

            # Truncate the input arrays to the requested size
            self.y = y_shuffled[:self.n_total]
            self.X_particles = X_particles_shuffled[:self.n_total]
            print(f'y_shuffled sum: {np.sum(self.y)}')
            print(f'y_shuffled shape: {self.y.shape}')

            # Also compute some jet observables
            self.compute_jet_observables()

            # Set up dict to store roc curves
            self.roc_curve_dict = {}
            self.roc_curve_dict_lasso = {}
            self.N_terms_lasso = {}
            self.observable_lasso = {}
            for model in self.models:
                self.roc_curve_dict[model] = {}

            # Plot the input data
            self.plot_QA()

            # Compute EFPs
            if 'efp_dnn' in self.models or 'efp_linear' in self.models or 'efp_lasso' in self.models:

                print()
                print(f'Calculating d <= {self.dmax} EFPs for {self.n_total} jets... ')
                
                # Specify parameters of EFPs
                # TODO: check beta dependence !!
                efpset = energyflow.EFPSet(('d<=', self.dmax), measure=self.efp_measure, beta=self.efp_beta)

                # Load labels and data, four vectors. Format: (pT,y,phi,m). 
                # Note: no PID yet which would be 5th entry... check later!
                # To make sure, don't need any further preprocessing like for EFNs?
                X_EFP = self.X_particles[:,:,:4] # Remove pid,charge from self.X_particles
                Y_EFP = self.y #Note not "to_categorical" here... 
    
                # Switch here to Jesse's quark/gluon data set.
                #X_EFP, self.Y_EFP = energyflow.datasets.qg_jets.load(self.n_train + self.n_val + self.n_test)
                
                # Convert to list of np.arrays of jets in format (pT,y,phi,mass or PID) -> dim: (# jets, # particles in jets, #4)
                # and remove zero entries
                masked_X_EFP = [x[x[:,0] > 0] for x in X_EFP]
                
                # Now compute EFPs
                X_EFP = efpset.batch_compute(masked_X_EFP)

                # Record which EFPs correspond to which indices
                # Note: graph images are available here: https://github.com/pkomiske/EnergyFlow/tree/images/graphs
                self.graphs = efpset.graphs()[1:]
                for i,efp in enumerate(self.graphs):
                    print(f'  efp {i} -- edges: {efp}')

                # Preprocess, plot, and store the EFPs for each d
                self.X_EFP_train = {}
                self.X_EFP_test = {}
                self.Y_EFP_train = {}
                self.Y_EFP_test = {}
                for d in range(1, self.dmax+1):

                    # Select EFPs with degree <= d
                    X_EFP_d = X_EFP[:,efpset.sel(('d<=', d))]

                    # Remove the 0th EFP (=1)
                    X_EFP_d = X_EFP_d[:,1:]
                    print(f'There are {X_EFP_d.shape[1]} terms for d<={d} (connected + disconnected, and excluding d=0)')

                    # Plot EFPs
                    if d == 2:
                        self.plot_efp_distributions(d, X_EFP_d, suffix='before_scaling')
                        self.plot_efp_distributions(d, sklearn.preprocessing.scale(X_EFP_d.astype(np.float128)), suffix='after_scaling')

                    # Do train/val/test split (Note: separate val_set generated in DNN training.)
                    (X_EFP_train_d, X_EFP_val, 
                        self.X_EFP_test[d], self.Y_EFP_train[d], 
                        Y_EFP_val, self.Y_EFP_test[d]) = energyflow.utils.data_split(X_EFP_d, Y_EFP, val=self.n_val, test=self.n_test)
                    
                    # Preprocessing: zero mean unit variance
                    self.X_EFP_train[d] = sklearn.preprocessing.scale(X_EFP_train_d.astype(np.float128))

                print('Done.') 

                # Plot a few single observables

                # EFP Lasso for paper -- run with d = 4
                #observable = '[(0, 1)] + 3.54* [(0, 1), (0, 2)] + 1.72 * [(0, 1), (0, 2), (0, 3), (0, 4)] -3.82 * [(0, 1), (0, 1), (2, 3), (2, 3)]'
                if self.dmax == 4:
                    observable = rf'$\mathcal{{O}}^{{\mathrm{{ML}}}}_{{\mathrm{{EFP}}}}$ (4 terms)' 
                    ylabel = rf'$\frac{{1}}{{\sigma}} \frac{{d\sigma}}{{ d \mathcal{{O}}^{{\mathrm{{ML}}}}_{{\mathrm{{EFP}}}} }}$'
                    X = self.X_EFP_train[self.dmax][:,0] + 3.54*self.X_EFP_train[self.dmax][:,4] + 1.72*self.X_EFP_train[self.dmax][:,17] - 3.82*self.X_EFP_train[self.dmax][:,23]
                    y = self.Y_EFP_train[self.dmax]
                    self.plot_observable(X, y, xlabel=observable, ylabel=ylabel, filename='EFP_0_4_17_23.pdf')

            # Train models
            self.train_models(jet_pt_min)

        # Run plotting script
        print()
        print('Run plotting script...')
        cmd = f'python analysis/plot_flavor.py -c {self.config_file} -o {self.output_dir}'
        subprocess.run(cmd, check=True, shell=True)

    #---------------------------------------------------------------
    # Parse the input file into a 3D array (jets, particles, particle info)
    # The particle info will be stored as: (pt, eta, phi, m, pid, charge)
    #---------------------------------------------------------------
    def create_jet_array(self, jet_df):

        # Add columns of mass and charge
        # Note that some PIDs are not recognized by the energyflow functions (311 -- K0)
        # TODO: check whether we want to write these as K0L/K0S (for now we set error_on_unknown=False)
        jet_df['m'] = energyflow.pids2ms(jet_df['pid'], error_on_unknown=False)
        jet_df['charge'] = energyflow.pids2chrgs(jet_df['pid'], error_on_unknown=False)

        # Switch order: (pt, eta, phi, pid, m, charge) --> (pt, eta, phi, m, pid, charge)
        columns = list(jet_df.columns)
        index_pid = columns.index('pid')
        index_mass = columns.index('m')
        columns[index_pid], columns[index_mass] = columns[index_mass], columns[index_pid]
        jet_df = jet_df[columns]

        # Kyle's IO: Get nested list of particle info for each jet, then convert to numpy array
        # TODO: This could be made more efficient by translating directly into numpy array rather than list
        g = jet_df.drop(columns=['qg','ct']).groupby('num').cumcount()                                                                                     
        L = (jet_df.drop(columns=['qg','ct']).set_index(['num',g])                                                                                         
            .unstack(fill_value=0)                                                                                                                       
            .stack().groupby(level=0)                                                                                                                    
                .apply(lambda x: x.values.tolist())                                                                                                         
                .tolist())    
        jet_array = np.array(L)

        return jet_array

    #---------------------------------------------------------------
    # Compute some individual jet observables
    # TODO: speed up w/numba
    #---------------------------------------------------------------
    def compute_jet_observables(self):

        self.qa_results = defaultdict(list)
        self.qa_observables = [f'jet_charge_k{kappa}' for kappa in self.kappa]

        # Compute jet charge
        for kappa in self.kappa:

            for jet in self.X_particles:

                jet_charge = 0
                jet_pt = 0
                for particle in jet:
                    pt = particle[0]
                    charge = particle[5]
                    jet_pt += pt
                    jet_charge += charge * np.power(pt, kappa)
                jet_charge = jet_charge / np.power(jet_pt, kappa)
                self.qa_results[f'jet_charge_k{kappa}'].append(jet_charge)

    #---------------------------------------------------------------
    # Train models
    #---------------------------------------------------------------
    def train_models(self, jet_pt_min):

        # Train ML models
        self.key_suffix = f'pt{jet_pt_min}'
        for model in self.models:
            print()
        
            # Dict to store AUC
            self.AUC[f'{model}{self.key_suffix}'] = []
        
            model_settings = self.model_settings[model]

            # EFPs
            for d in range(1, self.dmax+1):
                if model == 'efp_linear':
                    self.fit_efp_linear(model, model_settings, d)
                if model == 'efp_dnn':
                    self.fit_efp_dnn(model, model_settings, d)
            if model == 'efp_lasso':
                self.fit_efp_lasso(model, model_settings, self.d_lasso)

            # Deep sets
            if model == 'pfn':
                self.fit_pfn(model, model_settings, self.y, self.X_particles)
                
            if model == 'efn':
                self.fit_efn(model, model_settings)

        # Plot traditional observables
        for observable in self.qa_observables:
            self.roc_curve_dict_lasso[observable] = sklearn.metrics.roc_curve(self.y, -np.array(self.qa_results[observable]))
            self.roc_curve_dict[observable] = sklearn.metrics.roc_curve(self.y, -np.array(self.qa_results[observable]))

        # Save ROC curves to file
        if 'nsub_dnn' in self.models or 'efp_dnn' in self.models or 'nsub_linear' in self.models or 'efp_linear' in self.models or 'pfn' in self.models or 'efn' in self.models:
            output_filename = os.path.join(self.output_dir_i, f'ROC{self.key_suffix}.pkl')
            with open(output_filename, 'wb') as f:
                pickle.dump(self.roc_curve_dict, f)
                pickle.dump(self.AUC, f)

        # Separate lasso from others, so that we can re-run it quickly
        if 'efp_lasso' in self.models:
            output_filename = os.path.join(self.output_dir_i, f'ROC{self.key_suffix}_lasso.pkl')
            with open(output_filename, 'wb') as f_lasso:
                pickle.dump(self.roc_curve_dict_lasso, f_lasso)
                pickle.dump(self.N_terms_lasso, f_lasso)
                pickle.dump(self.observable_lasso, f_lasso)

    #---------------------------------------------------------------
    # Fit linear model for EFPs
    #---------------------------------------------------------------
    def fit_efp_linear(self, model, model_settings, d):

        X_train = self.X_EFP_train[d]
        X_test = self.X_EFP_test[d]
        y_train = self.Y_EFP_train[d]
        y_test = self.Y_EFP_test[d]
        self.fit_linear_model(X_train, y_train, X_test, y_test, model, model_settings, dim_label='d', dim=d, type='LDA_search')

    #---------------------------------------------------------------
    # Fit Lasso for EFPs
    #---------------------------------------------------------------
    def fit_efp_lasso(self, model, model_settings, d):

        X_train = self.X_EFP_train[self.d_lasso]
        X_test = self.X_EFP_test[self.d_lasso]
        y_train = self.Y_EFP_train[self.d_lasso]
        y_test = self.Y_EFP_test[self.d_lasso]
        self.fit_lasso(X_train, y_train, X_test, y_test, model, model_settings, dim_label='d', dim=d, observable_type='sum')

    #---------------------------------------------------------------
    # Fit Dense Neural Network for EFPs
    #---------------------------------------------------------------
    def fit_efp_dnn(self, model, model_settings, d):

        X_train = self.X_EFP_train[d]
        X_test = self.X_EFP_test[d]
        y_train = self.Y_EFP_train[d]
        y_test = self.Y_EFP_test[d]
        self.fit_dnn(X_train, y_train, X_test, y_test, model, model_settings, dim_label='d', dim=d)

    #---------------------------------------------------------------
    # Fit ML model -- SGDClassifier or LinearDiscriminant
    #   - SGDClassifier: Linear model (SVM by default, w/o kernel) with SGD training
    #   - For best performance, data should have zero mean and unit variance
    #---------------------------------------------------------------
    def fit_linear_model(self, X_train, y_train, X_test, y_test, model, model_settings, dim_label='', dim=None, type='SGD'):
        print(f'Training {model} ({type}), {dim_label}={dim}...')
        
        if type == 'SGD':
        
            # Define model
            clf = sklearn.linear_model.SGDClassifier(loss=model_settings['sgd_loss'],
                                                        max_iter=model_settings['sgd_max_iter'],
                                                        learning_rate=model_settings['sgd_learning_rate'],
                                                        early_stopping=model_settings['sgd_early_stopping'],
                                                        random_state=self.random_state)

            # Optimize hyperparameters with random search, using cross-validation to determine best set
            # Here we just search over discrete values, although can also easily specify a distribution
            param_distributions = {'penalty': model_settings['sgd_penalty'],
                                'alpha': model_settings['sgd_alpha'],
                                'tol': model_settings['sgd_tol']}

            randomized_search = sklearn.model_selection.RandomizedSearchCV(clf, param_distributions,
                                                                           n_iter=model_settings['n_iter'],
                                                                           cv=model_settings['cv'],
                                                                           random_state=self.random_state)
            search_result = randomized_search.fit(X_train, y_train)
            final_model = search_result.best_estimator_
            result_info = search_result.cv_results_
            print(f'Best params: {search_result.best_params_}')

            # Get predictions for the test set
            #y_predict_train = final_model.predict(X_train)
            #y_predict_test = final_model.predict(X_test)
            
            y_predict_train = sklearn.model_selection.cross_val_predict(clf, X_train, y_train, cv=3, method="decision_function")
            
            # Compare AUC on train set and test set
            AUC_train = sklearn.metrics.roc_auc_score(y_train, y_predict_train)
            print(f'AUC = {AUC_train} (cross-val train set)')
            print()

            # Compute ROC curve: the roc_curve() function expects labels and scores
            self.roc_curve_dict[model][dim] = sklearn.metrics.roc_curve(y_train, y_predict_train)
        
            # Check number of threhsolds used for ROC curve
            # print('thresholds: {}'.format(self.roc_curve_dict[model][K][2]))
            
            # Plot confusion matrix
            #self.plot_confusion_matrix(self.y_train, y_predict_train, f'{model}_K{K}')

        elif type == 'LDA':

            # energyflow implementation
            clf = energyflow.archs.LinearClassifier(linclass_type='lda')
            history = clf.fit(X_train, y_train)
            preds_EFP = clf.predict(X_test)        
            auc_EFP = sklearn.metrics.roc_auc_score(y_test,preds_EFP[:,1])
            print(f'  AUC = {auc_EFP} (test set)')
            self.AUC[f'{model}{self.key_suffix}'].append(auc_EFP)
            self.roc_curve_dict[model][dim] = sklearn.metrics.roc_curve(y_test, preds_EFP[:,1])

        elif type == 'LDA_search':

            # Define model
            clf = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()

            # Optimize hyperparameters
            param_distributions = {'tol': model_settings['lda_tol']}

            randomized_search = sklearn.model_selection.GridSearchCV(clf, param_distributions)
            search_result = randomized_search.fit(X_train, y_train)
            final_model = search_result.best_estimator_
            result_info = search_result.cv_results_
            print(f'Best params: {search_result.best_params_}')

            y_predict_train = sklearn.model_selection.cross_val_predict(clf, X_train, y_train, cv=3, method="decision_function")
            
            # Compare AUC on train set and test set
            AUC_train = sklearn.metrics.roc_auc_score(y_train, y_predict_train)
            print(f'AUC = {AUC_train} (cross-val train set)')
            print()

            # Compute ROC curve: the roc_curve() function expects labels and scores
            self.roc_curve_dict[model][dim] = sklearn.metrics.roc_curve(y_train, y_predict_train)

    #---------------------------------------------------------------
    # Fit Lasso
    #
    #   The parameter alpha multiplies to L1 term
    #   If convergence error: can increase max_iter and/or tol, and/or set normalize=True
    # 
    # observable_type: ['product', 'sum']
    #   - if sum observable, we assume preprocessing is done beforehand
    #   - if product observable, can uncomment preprocessing below after taking log (but off by default)
    #---------------------------------------------------------------
    def fit_lasso(self, X_train, y_train, X_test, y_test, model, model_settings, dim_label='', dim=None, observable_type='product'):
        print()
        print(f'Training {model} ({observable_type} observable), {dim_label}={dim}...')
        
        # First, copy the test training labels, which we will need for ROC curve
        # This is needed because for product observable we don't want to take the log
        y_test_roc = y_test.copy()
        y_train_roc = y_train.copy()

        # If product observable, take the logarithm of the data and labels, such that the product observable 
        # becomes a sum and the exponents in the product observable become the regression weights
        if observable_type == 'product':

            offset = 1.e-4
            X_train = np.log(X_train + offset)
            X_test = np.log(X_test + offset)

            eps = .01
            y_train = np.log(eps + (1. - 2. * eps) * y_train)
            y_test = np.log(eps + (1. - 2. * eps) * y_test)

            # Preprocessing: zero mean unit variance
            #X_train = sklearn.preprocessing.scale(X_train)
            #X_test = sklearn.preprocessing.scale(X_test)

        # Loop through values of regularization parameter
        self.roc_curve_dict_lasso[model] = {}
        self.N_terms_lasso[model] = {}
        self.observable_lasso[model] = {}

        for alpha in model_settings['alpha']:
            self.fit_lasso_single_alpha(alpha, X_train, y_train, X_test, y_test, y_test_roc, y_train_roc, model, model_settings, 
                                        dim_label=dim_label, dim=dim, observable_type=observable_type)

    #---------------------------------------------------------------
    # Fit Lasso for a single alpha value
    #---------------------------------------------------------------
    def fit_lasso_single_alpha(self, alpha, X_train, y_train, X_test, y_test, y_test_roc, y_train_roc, model, model_settings, 
                                dim_label='', dim=None, observable_type='product'):
        print()
        print(f'Fitting lasso regression with alpha = {alpha}')
    
        lasso_clf = sklearn.linear_model.Lasso(alpha=alpha, max_iter=model_settings['max_iter'],
                                                tol=model_settings['tol'])
                                                
        plot_learning_curve = False
        if plot_learning_curve:
            # Split into validation set
            X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X_train, y_train, test_size=0.2)
            train_errors = []
            validation_errors = []
            train_sizes = np.linspace(0, len(X_train)/2, 50)[1:]
            print('Compute Lasso learning curve...')
            for train_size in train_sizes:
                train_size = int(train_size)
                lasso_clf.fit(X_train[:train_size], y_train[:train_size])
                y_predict_train = lasso_clf.predict(X_train[:train_size])
                y_predict_val = lasso_clf.predict(X_val)
                train_errors.append(sklearn.metrics.mean_squared_error(y_predict_train, y_train[:train_size]))
                validation_errors.append(sklearn.metrics.mean_squared_error(y_predict_val, y_val))
        else:
            # Cross-validation
            lasso_clf.fit(X_train, y_train)
            scores = sklearn.model_selection.cross_val_score(lasso_clf, X_train, y_train,
                                                                        scoring='neg_mean_squared_error',
                                                                        cv=model_settings['cv'])
            print(f'cross-validation scores: {scores}')
            y_predict_train = lasso_clf.predict(X_train)
            rmse = sklearn.metrics.mean_squared_error(y_train, y_predict_train)
            print(f'training rmse: {rmse}')

        # Compute AUC on test set
        y_predict_test = lasso_clf.predict(X_test)
        auc_test = sklearn.metrics.roc_auc_score(y_test_roc, y_predict_test)
        rmse_test = sklearn.metrics.mean_squared_error(y_test, y_predict_test)
        print(f'AUC = {auc_test} (test set)')
        print(f'test rmse: {rmse_test}')
        
        # ROC curve
        self.roc_curve_dict_lasso[model][alpha] = sklearn.metrics.roc_curve(y_test_roc, y_predict_test)
        
        if plot_learning_curve:
            plt.axis([0, train_sizes[-1], 0, 10])
            plt.xlabel('training size', fontsize=16)
            plt.ylabel('MSE', fontsize=16)
            plt.plot(train_sizes, train_errors, linewidth=2,
                        linestyle='solid', alpha=0.9, color=sns.xkcd_rgb['dark sky blue'], label='train')
            plt.plot(train_sizes, validation_errors, linewidth=2,
                        linestyle='solid', alpha=0.9, color=sns.xkcd_rgb['watermelon'], label='val')
            plt.axline((0, rmse_test), (len(X_train), rmse_test), linewidth=4, label='test',
                        linestyle='dotted', alpha=0.9, color=sns.xkcd_rgb['medium green'])
            plt.legend(loc='best', fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir_i, f'Lasso_learning_curve_a{alpha}.pdf'))
            plt.close()

        # Print out observable
        observable = ''
        n_terms = 0
        coeffs = lasso_clf.coef_
        nonzero_coeffs = coeffs[np.absolute(coeffs)>1e-10]
        mean_coeff = np.mean(np.absolute(nonzero_coeffs))
        if observable_type == 'product' and np.mean(nonzero_coeffs) < 0.:
            mean_coeff *= -1
        elif observable_type == 'sum' and np.mean( np.dot(X_test, coeffs) ) < 0:
            mean_coeff *= -1
        print(f'mean_coeff: {mean_coeff}')
        coeffs = np.divide(coeffs, mean_coeff)
        for i,_ in enumerate(coeffs):
            coeff = np.round(coeffs[i], 3)
            if not np.isclose(coeff, 0., atol=1e-10):
                n_terms += 1

                if observable_type == 'product':
                    if 'efp' in model:
                        observable += rf'({self.graphs[i]})^{{{coeff}}} '

                elif observable_type == 'sum':
                    if 'efp' in model:
                        if n_terms > 0:
                            observable += ' + '
                        observable += f'{coeff} * {self.graphs[i]}'

        print(f'Observable: {observable}')

        self.N_terms_lasso[model][alpha] = n_terms
        self.observable_lasso[model][alpha] = observable

        # Plot observable
        if observable_type == 'product':
            designed_observable = np.exp( np.dot(X_train, coeffs) )
            y = y_train_roc
        elif observable_type == 'sum':
            designed_observable = np.dot(X_test, coeffs) # For EFPs use X_test since is not preprocessed
            y = y_test_roc
        xlabel = rf'$\mathcal{{O}} = {observable}$'
        ylabel = rf'$\frac{{1}}{{\sigma}} \frac{{d\sigma}}{{ d \mathcal{{O}} }}$'
        if 'nsub' in model:
            if n_terms < 5:
                xfontsize=12
            else:
                xfontsize=8
        elif 'efp' in model:
            if n_terms < 2:
                xfontsize=12
            else:
                xfontsize=6
        logy = 'nsub' in model

        self.plot_observable(designed_observable, y, xlabel=xlabel, ylabel=ylabel, filename=f'{model}_{alpha}.pdf', logy=logy)

    #---------------------------------------------------------------
    # Train DNN, using hyperparameter optimization with keras tuner
    #---------------------------------------------------------------
    def fit_dnn(self, X_train, Y_train, X_test, Y_test, model, model_settings, dim_label='', dim=None):
        print()
        print(f'Training {model}, {dim_label}={dim}...')

        tuner = keras_tuner.Hyperband(functools.partial(self.dnn_builder, input_shape=[X_train.shape[1]], model_settings=model_settings),
                                        objective='val_accuracy',
                                        max_epochs=10,
                                        factor=3,
                                        directory='keras_tuner',
                                        project_name=f'{model}{dim}')

        tuner.search(X_train, Y_train, 
                        batch_size=model_settings['batch_size'],
                        epochs=model_settings['epochs'], 
                        validation_split=self.val_frac)
        
        # Get the optimal hyperparameters
        best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
        units1 = best_hps.get('units1')
        units2 = best_hps.get('units2')
        units3 = best_hps.get('units3')
        learning_rate = best_hps.get('learning_rate')
        print()
        print(f'Best hyperparameters:')
        print(f'   units: ({units1}, {units2}, {units3})')
        print(f'   learning_rate: {learning_rate}')
        print()

        # Retrain the model with best number of epochs
        hypermodel = tuner.hypermodel.build(best_hps)
        history = hypermodel.fit(X_train, Y_train, epochs=model_settings['epochs'], validation_split=self.val_frac)

        # Plot metrics as a function of epochs
        self.plot_NN_epochs(model_settings['epochs'], history, model, dim_label=dim_label, dim=dim) 

        # Get predictions for test data set
        preds_DNN = hypermodel.predict(X_test).reshape(-1)
        
        # Get AUC
        auc_DNN = sklearn.metrics.roc_auc_score(Y_test, preds_DNN)
        print(f'  AUC = {auc_DNN} (test set)')
        
        # Store AUC
        self.AUC[f'{model}{self.key_suffix}'].append(auc_DNN)
        
        # Get & store ROC curve
        self.roc_curve_dict[model][dim] = sklearn.metrics.roc_curve(Y_test, preds_DNN)

    #---------------------------------------------------------------
    # Construct model for hyperparameter tuning with keras tuner
    #---------------------------------------------------------------
    def dnn_builder(self, hp, input_shape, model_settings):

        model = keras.models.Sequential()
        model.add(keras.layers.Flatten(input_shape=input_shape))

        # Tune size of first dense layer
        hp_units1 = hp.Int('units1', min_value=32, max_value=512, step=32)
        hp_units2 = hp.Int('units2', min_value=32, max_value=512, step=32)
        hp_units3 = hp.Int('units3', min_value=32, max_value=512, step=32)
        model.add(keras.layers.Dense(units=hp_units1, activation='relu'))
        model.add(keras.layers.Dense(units=hp_units2, activation='relu'))
        model.add(keras.layers.Dense(units=hp_units3, activation='relu'))
        model.add(keras.layers.Dense(1,activation='sigmoid'))  # softmax? # Last layer has to be 1 or 2 for binary classification?

        # Print DNN summary
        model.summary()

        # Tune the learning rate for the optimizer
        hp_learning_rate = hp.Choice('learning_rate', values=model_settings['learning_rate']) # if error, change name to lr or learning_rate

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),  # For Stochastic gradient descent use: SGD
                      loss=model_settings['loss'],
                      metrics=model_settings['metrics'])

        return model

    #---------------------------------------------------------------
    # Fit ML model -- Deep Set/Particle Flow Networks
    #---------------------------------------------------------------
    def fit_pfn(self, model, model_settings, y, X_particles):
    
        # Convert labels to categorical
        Y_PFN = energyflow.utils.to_categorical(y, num_classes=2)
                        
        # (pT,y,phi,pid)
        X_PFN = X_particles[:,:,[0,1,2,4]]

        # Preprocess by centering jets and normalizing pts
        for x_PFN in X_PFN:
            mask = x_PFN[:,0] > 0
            yphi_avg = np.average(x_PFN[mask,1:3], weights=x_PFN[mask,0], axis=0)
            x_PFN[mask,1:3] -= yphi_avg
            x_PFN[mask,0] /= x_PFN[:,0].sum()
        
        # Handle particle id channel
        #if model_settings['use_pids']:
        #    self.my_remap_pids(X_PFN)
        #else:
        X_PFN = X_PFN[:,:,:3]
        
        # Check shape
        if y.shape[0] != X_PFN.shape[0]:
            print(f'Number of labels {y.shape} does not match number of jets {X_PFN.shape} ! ')

        # Split data into train, val and test sets
        (X_PFN_train, X_PFN_val, X_PFN_test, Y_PFN_train, Y_PFN_val, Y_PFN_test) = energyflow.utils.data_split(X_PFN, Y_PFN,
                                                                                             val=self.n_val, test=self.n_test)
        # Build architecture
        pfn = energyflow.archs.PFN(input_dim=X_PFN.shape[-1],
                                   Phi_sizes=model_settings['Phi_sizes'],
                                   F_sizes=model_settings['F_sizes'])

        # Train model
        history = pfn.fit(X_PFN_train,
                          Y_PFN_train,
                          epochs=model_settings['epochs'],
                          batch_size=model_settings['batch_size'],
                          validation_data=(X_PFN_val, Y_PFN_val),
                          verbose=1)
                          
        # Plot metrics are a function of epochs
        self.plot_NN_epochs(model_settings['epochs'], history, model)
        
        # Get predictions on test data
        preds_PFN = pfn.predict(X_PFN_test, batch_size=1000)

        # Get AUC and ROC curve + make plot
        auc_PFN = sklearn.metrics.roc_auc_score(Y_PFN_test[:,1], preds_PFN[:,1])
        print('Particle Flow Networks/Deep Sets: AUC = {} (test set)'.format(auc_PFN))
        self.AUC[f'{model}{self.key_suffix}'].append(auc_PFN)
        
        self.roc_curve_dict[model] = sklearn.metrics.roc_curve(Y_PFN_test[:,1], preds_PFN[:,1])
        
    #---------------------------------------------------------------
    # Fit ML model -- (IRC safe) Energy Flow Networks
    #---------------------------------------------------------------
    def fit_efn(self, model, model_settings):
    
        # Convert labels to categorical
        Y_EFN = energyflow.utils.to_categorical(self.y, num_classes=2)
                        
        # (pT,y,phi,m)
        X_EFN = self.X_particles[:,:,:4] # Remove pid,charge from self.X_particles
        
        # Can switch here to quark vs gluon data set
        #X_EFN, y_EFN = energyflow.datasets.qg_jets.load(self.n_train + self.n_val + self.n_test)
        #Y_EFN = energyflow.utils.to_categorical(y_EFN, num_classes=2)
        #print('(n_jets, n_particles per jet, n_variables): {}'.format(X_EFN.shape))

        # For now just use the first 30 entries of the 4-vectors per jet (instead of 800)
        #np.set_printoptions(threshold=sys.maxsize)        
        #X_EFN = X_EFN[:,:30]        
        
        # Preprocess data set by centering jets and normalizing pts
        # Note: this step is somewhat different for pp/AA compared to the quark/gluon data set
        for x_EFN in X_EFN:
            mask = x_EFN[:,0] > 0
            
            # Compute y,phi averages
            yphi_avg = np.average(x_EFN[mask,1:3], weights=x_EFN[mask,0], axis=0)

            # Adjust phi range: Initially it is [0,2Pi], now allow for negative values and >2Pi 
            # so there are no gaps for a given jet.
            # Mask particles that are far away from the average phi & cross the 2Pi<->0 boundary
            mask_phi_1 = ((x_EFN[:,2] - yphi_avg[1] >  np.pi) & (x_EFN[:,2] != 0.))
            mask_phi_2 = ((x_EFN[:,2] - yphi_avg[1] < -np.pi) & (x_EFN[:,2] != 0.))
            
            x_EFN[mask_phi_1,2] -= 2*np.pi
            x_EFN[mask_phi_2,2] += 2*np.pi            
            
            # Now recompute y,phi averages after adjusting the phi range
            yphi_avg1 = np.average(x_EFN[mask,1:3], weights=x_EFN[mask,0], axis=0)            
            
            # And center jets in the y,phi plane
            x_EFN[mask,1:3] -= yphi_avg1

            # Normalize transverse momenta p_Ti -> z_i
            x_EFN[mask,0] /= x_EFN[:,0].sum()
            
            # Set particle four-vectors to zero if the z value is below a certain threshold.
            mask2 = x_EFN[:,0]<0.00001
            x_EFN[mask2,:]=0
        
        # Do not use PID for EFNs
        X_EFN = X_EFN[:,:,:3]
        
        # Make 800 four-vector array smaller, e.g. only 150. Ok w/o background
        X_EFN = X_EFN[:,:150]
        
        # Check shape
        if self.y.shape[0] != X_EFN.shape[0]:
            print(f'Number of labels {self.y.shape} does not match number of jets {X_EFN.shape} ! ')
            
        # Split data into train, val and test sets 
        # and separate momentum fraction z and angles (y,phi)
        (z_EFN_train, z_EFN_val, z_EFN_test, 
         p_EFN_train, p_EFN_val, p_EFN_test,
         Y_EFN_train, Y_EFN_val, Y_EFN_test) = energyflow.utils.data_split(X_EFN[:,:,0], X_EFN[:,:,1:], Y_EFN, 
                                                                           val=self.n_val, test=self.n_test)
        
        # Build architecture
        opt = keras.optimizers.Adam(learning_rate=model_settings['learning_rate']) # if error, change name to learning_rate
        efn = energyflow.archs.EFN(input_dim=2,
                                   Phi_sizes=model_settings['Phi_sizes'],
                                   F_sizes=model_settings['F_sizes'],
                                   optimizer=opt)
        
        # Train model
        history = efn.fit([z_EFN_train,p_EFN_train],
                          Y_EFN_train,
                          epochs=model_settings['epochs'],
                          batch_size=model_settings['batch_size'],
                          validation_data=([z_EFN_val,p_EFN_val], Y_EFN_val),
                          verbose=1)
                          
        # Plot metrics are a function of epochs
        self.plot_NN_epochs(model_settings['epochs'], history, model)
        
        # Get predictions on test data
        preds_EFN = efn.predict([z_EFN_test,p_EFN_test], batch_size=1000)     

        # Get AUC and ROC curve + make plot
        auc_EFN = sklearn.metrics.roc_auc_score(Y_EFN_test[:,1], preds_EFN[:,1])
        print('(IRC safe) Energy Flow Networks: AUC = {} (test set)'.format(auc_EFN))
        self.AUC[f'{model}{self.key_suffix}'].append(auc_EFN)
        
        self.roc_curve_dict[model] = sklearn.metrics.roc_curve(Y_EFN_test[:,1], preds_EFN[:,1])

    #--------------------------------------------------------------- 
    # My own remap PID routine (similar to remap_pids from energyflow)
    #---------------------------------------------------------------         
    def my_remap_pids(self,events, pid_i=3, error_on_unknown=True):
        # PDGid to small float dictionary
        PID2FLOAT_MAP = {0: 0.0, 22: 1.4,
                         211: .1, -211: .2,
                         321: .3, -321: .4,
                         130: .5,
                         2112: .6, -2112: .7,
                         2212: .8, -2212: .9,
                         11: 1.0, -11: 1.1,
                         13: 1.2, -13: 1.3}
        
        """Remaps PDG id numbers to small floats for use in a neural network.
        `events` are modified in place and nothing is returned.
    
        **Arguments**
    
        - **events** : _numpy.ndarray_
            - The events as an array of arrays of particles.
        - **pid_i** : _int_
            - The column index corresponding to pid information in an event.
        - **error_on_unknown** : _bool_
            - Controls whether a `KeyError` is raised if an unknown PDG ID is
            encountered. If `False`, unknown PDG IDs will map to zero.
        """
    
        if events.ndim == 3:
            pids = events[:,:,pid_i].astype(int).reshape((events.shape[0]*events.shape[1]))
            if error_on_unknown:
                events[:,:,pid_i] = np.asarray([PID2FLOAT_MAP[pid]
                                                for pid in pids]).reshape(events.shape[:2])
            else:
                events[:,:,pid_i] = np.asarray([PID2FLOAT_MAP.get(pid, 0)
                                                for pid in pids]).reshape(events.shape[:2])
        else:
            if error_on_unknown:
                for event in events:
                    event[:,pid_i] = np.asarray([PID2FLOAT_MAP[pid]
                                                 for pid in event[:,pid_i].astype(int)])
            else:
                for event in events:
                    event[:,pid_i] = np.asarray([PID2FLOAT_MAP.get(pid, 0)
                                                 for pid in event[:,pid_i].astype(int)])        

    #---------------------------------------------------------------
    # Plot NN metrics are a function of epochs
    #---------------------------------------------------------------
    def plot_NN_epochs(self, n_epochs, history, label, dim_label='', dim=None):
    
        epoch_list = range(1, n_epochs+1)
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        if 'acc' in history.history:
            acc = history.history['acc']
            val_acc = history.history['val_acc']
        else:
            acc = history.history['accuracy']
            val_acc = history.history['val_accuracy']
        
        plt.axis([0, n_epochs, 0, 1])
        plt.xlabel('epochs', fontsize=16)
        plt.plot(epoch_list, loss, linewidth=2,
                 linestyle='solid', alpha=0.9, color=sns.xkcd_rgb['dark sky blue'],
                 label='loss')
        plt.plot(epoch_list, val_loss, linewidth=2,
                 linestyle='solid', alpha=0.9, color=sns.xkcd_rgb['faded purple'],
                 label='val_loss')
        plt.plot(epoch_list, acc, linewidth=2,
                 linestyle='dotted', alpha=0.9, color=sns.xkcd_rgb['watermelon'],
                 label='acc')
        plt.plot(epoch_list, val_acc, linewidth=2,
                 linestyle='dotted', alpha=0.9, color=sns.xkcd_rgb['medium green'],
                 label='val_acc')
        
        plt.legend(loc='best', fontsize=12)
        plt.tight_layout()
        if dim:
            plt.savefig(os.path.join(self.output_dir_i, f'DNN_epoch_{label}_{dim_label}{dim}.pdf'))
        else:
            plt.savefig(os.path.join(self.output_dir_i, f'PFN_epoch_{label}.pdf'))
        plt.close()

    #---------------------------------------------------------------
    # Plot confusion matrix
    # Note: not normalized to relative error
    #---------------------------------------------------------------
    def plot_confusion_matrix(self, y_train, y_predict_train, label):
    
        confusion_matrix = sklearn.metrics.confusion_matrix(y_train, y_predict_train)
        sns.heatmap(confusion_matrix)
        plt.savefig(os.path.join(self.output_dir_i, f'confusion_matrix_{label}.pdf'))
        plt.close()
        
    #---------------------------------------------------------------
    # Plot QA
    #---------------------------------------------------------------
    def plot_QA(self):
    
        for qa_observable in self.qa_observables:

            result = np.array(self.qa_results[qa_observable])
            if self.y.shape[0] != len(result):
                sys.exit(f'ERROR: {qa_observable}: {len(result)}, y shape: {self.y.shape}')
               
            q_indices = self.y
            g_indices = 1 - self.y
            result_q = result[q_indices.astype(bool)]
            result_g = result[g_indices.astype(bool)]

            # Set some labels
            if 'jet_charge' in qa_observable:
                xlabel = rf'$Q_{{\kappa}}$'
                ylabel = rf'$\frac{{1}}{{\sigma}} \frac{{d\sigma}}{{dQ_{{\kappa}} }}$'
                bins = np.linspace(-1, 1., 100)
            else:
                ylabel = ''
                xlabel = rf'{qa_observable}'
                bins = np.linspace(0, np.amax(result_g), 20)
            plt.xlabel(xlabel, fontsize=14)
            plt.ylabel(ylabel, fontsize=16)

            if qa_observable == 'jet_pt':
                stat='count'
            else:
                stat='density'

            # Construct dataframes for histplot
            df_q = pd.DataFrame(result_q, columns=[xlabel])
            df_g = pd.DataFrame(result_g, columns=[xlabel])
            
            # Add label columns to each df to differentiate them for plotting
            df_q['generator'] = np.repeat(self.q_label, result_q.shape[0])
            df_g['generator'] = np.repeat(self.g_label, result_g.shape[0])
            df = df_q.append(df_g, ignore_index=True)

            # Histplot
            h = sns.histplot(df, x=xlabel, hue='generator', stat=stat, bins=bins, element='step', common_norm=False)
            h.legend_.set_title(None)
            plt.setp(h.get_legend().get_texts(), fontsize='14') # for legend text

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir_i, f'{qa_observable}.pdf'))
            plt.close()

    #---------------------------------------------------------------
    # Plot q vs. g
    #---------------------------------------------------------------
    def plot_observable(self, X, y_train, xlabel='', ylabel='', filename='', xfontsize=12, yfontsize=16, logx=False, logy=False):
            
        q_indices = y_train
        g_indices = 1 - y_train

        observable_q = X[q_indices.astype(bool)]
        observable_g = X[g_indices.astype(bool)]

        df_q = pd.DataFrame(observable_q, columns=[xlabel])
        df_g = pd.DataFrame(observable_g, columns=[xlabel])

        df_q['generator'] = np.repeat(self.AA_label, observable_q.shape[0])
        df_g['generator'] = np.repeat(self.pp_label, observable_g.shape[0])
        df = df_q.append(df_g, ignore_index=True)

        if filename == 'tau_10_11_14_14.pdf':
            #bins = 10 ** np.linspace(np.log10(1.e-16), np.log10(1.e-10), 50)
            bins = np.linspace(0., 1.e-11, 100)
            #bins = np.linspace(np.amin(X), np.amax(X), 50)
            print(df.describe())
            stat='count'
        else:
            bins = np.linspace(np.amin(X), np.amax(X), 50)
            stat='density'
        h = sns.histplot(df, x=xlabel, hue='generator', stat=stat, bins=bins, element='step', common_norm=False, log_scale=[False, logy])
        if h.legend_:
            #h.legend_.set_bbox_to_anchor((0.85, 0.85))
            h.legend_.set_title(None)
            plt.setp(h.get_legend().get_texts(), fontsize='14') # for legend text

        plt.xlabel(xlabel, fontsize=xfontsize)
        plt.ylabel(ylabel, fontsize=yfontsize)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir_i, f'{filename}'))
        plt.close()

    #---------------------------------------------------------------
    # Plot EFPs
    #---------------------------------------------------------------
    def plot_efp_distributions(self, d, X_EFP_d, suffix='before_scaling'):
            
        print(f'Plotting input EFP data {suffix}, d={d}...')

        # Separate q/g
        q_indices = self.y
        g_indices = 1 - self.y
        X_q = X_EFP_d[q_indices.astype(bool)]
        X_g = X_EFP_d[g_indices.astype(bool)]

        # Get labels
        graphs = [str(x) for x in self.graphs[:4]]

        # Construct dataframes for scatter matrix plotting
        df_q = pd.DataFrame(X_q, columns=graphs)
        df_g = pd.DataFrame(X_g, columns=graphs)
        
        # Add label columns to each df to differentiate them for plotting
        df_q['generator'] = np.repeat(self.g_label, X_q.shape[0])
        df_g['generator'] = np.repeat(self.q_label, X_g.shape[0])
        df = df_q.append(df_g, ignore_index=True)

        # Plot scatter matrix
        g = sns.pairplot(df, corner=True, hue='generator', plot_kws={'alpha':0.1})
        #g.legend.set_bbox_to_anchor((0.75, 0.75))
        #plt.savefig(os.path.join(self.output_dir_i, f'training_data_scatter_matrix_K{K}.png'), dpi=50)
        plt.savefig(os.path.join(self.output_dir_i, f'training_data_scatter_matrix_d{d}_{suffix}.pdf'))
        plt.close()
        
        # Plot correlation matrix
        df.drop(columns=['generator'])
        corr_matrix = df.corr()
        sns.heatmap(corr_matrix)
        plt.savefig(os.path.join(self.output_dir_i, f'training_data_correlation_matrix_d{d}_{suffix}.pdf'))
        plt.close()
            
##################################################################
if __name__ == '__main__':

    # Define arguments
    parser = argparse.ArgumentParser(description='Process qg')
    parser.add_argument('-c', '--configFile', action='store',
                        type=str, metavar='configFile',
                        default='config/qg.yaml',
                        help='Path of config file for analysis')
    parser.add_argument('-o', '--outputDir', action='store',
                        type=str, metavar='outputDir',
                        default='./TestOutput',
                        help='Output directory for output to be written to')

    # Parse the arguments
    args = parser.parse_args()

    print('Configuring...')
    print('configFile: \'{0}\''.format(args.configFile))
    print('ouputDir: \'{0}\"'.format(args.outputDir))

    # If invalid configFile is given, exit
    if not os.path.exists(args.configFile):
        print('File \"{0}\" does not exist! Exiting!'.format(args.configFile))
        sys.exit(0)

    analysis = AnalyzePPAA(config_file=args.configFile, output_dir=args.outputDir)
    analysis.analyze_pp_aa()