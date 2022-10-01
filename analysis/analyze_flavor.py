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
from particle import Particle
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
    n_particles = X_particles.shape[1]
    
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
class AnalyzeFlavor(common_base.CommonBase):

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

        self.event_type = config['event_type']
        self.classification_type = config['classification_type']
        self.classes = config['classes']

        if self.classification_type == 'jet':

            if self.event_type == 'photoproduction':
                self.class_map = {'q': 1,
                                  'g': 2
                                 }
            elif self.event_type == 'dis':
                self.class_map = {'anti-b': -5,
                                  'anti-c': -4,
                                  'anti-s': -3,
                                  'anti-u': -2,
                                  'anti-d': -1,
                                  'd': 1,
                                  'u': 2,
                                  's': 3,
                                  'c': 4,
                                  'b': 5
                                 }

        elif self.classification_type == 'event':

            if self.event_type == 'photoproduction':
                # See table 16: https://pythia.org/download/pdf/lutp0613man2.pdf
                self.class_map = {'qq --> qq': 11,
                                  'qqbar --> qqbar': 12,
                                  'qqbar --> gg': 13,
                                  'gg --> qqbar': 53,
                                  'gg --> gg': 68,
                                  'qg --> qg': 28,
                                  'LO DIS': 99,                # 'gamma* q --> q'
                                  'transverse QCDC': 131,      # 'gamma*T q --> qg'
                                  'longitudinal QCDC': 132,    # 'gamma*L q --> qg'
                                  'transverse PGF': 135,       # 'gamma*T g --> qqbar'
                                  'longitudinal PGF': 136      # 'gamma*L g --> qqbar'
                                 }
            elif self.event_type == 'dis':
                sys.exit(f'ERROR: event classification not implemented for DIS events')

        # Set the class labels based on the class type
        class_labels = self.classes.split('__')
        self.class1_label = class_labels[0]
        self.class2_label = class_labels[1]
        self.classes_class1 = self.class1_label.split('_')
        self.classes_class2 = self.class2_label.split('_')
        self.class1_ids = [self.class_map[class_i] for class_i in self.classes_class1]
        self.class2_ids = [self.class_map[class_i] for class_i in self.classes_class2]
        for class_i in self.classes_class1:
            if class_i not in self.class_map.keys():
                sys.exit(f'Class ({class_i}) not supported -- available classes are: {self.class_map.keys()}')
        for class_i in self.classes_class2:
            if class_i not in self.class_map.keys():
                sys.exit(f'Class ({class_i}) not supported -- available classes are: {self.class_map.keys()}')
        
        if self.classification_type == 'event':
            self.n_jets_max = config['n_jets_max']
            self.n_particles_per_jet_max = config['n_particles_per_jet_max']

        self.jet_pt_min_list= config['jet_pt_min_list']
        self.jetR = 0.4
        self.kappa = config['kappa']
        self.particle_pt_min_list = config['particle_pt_min_list']

        self.input_files = config['input_files']

        self.n_train = config['n_train']
        self.n_val = config['n_val']
        self.n_test = config['n_test']
        self.n_total = self.n_train + self.n_val + self.n_test
        self.train_frac = self.n_train / self.n_total
        self.test_frac = 1. * self.n_test / self.n_total
        self.val_frac = 1. * self.n_val / (self.n_train + self.n_val)
        self.balance_samples = config['balance_samples']
        
        if 'dmax' in config:
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
                self.model_settings[model]['pid'] = config[model]['pid']
                self.model_settings[model]['nopid'] = config[model]['nopid']
                self.model_settings[model]['charge'] = config[model]['charge']
                
            if model == 'efn':
                self.model_settings[model]['Phi_sizes'] = tuple(config[model]['Phi_sizes'])
                self.model_settings[model]['F_sizes'] = tuple(config[model]['F_sizes'])
                self.model_settings[model]['epochs'] = config[model]['epochs']
                self.model_settings[model]['batch_size'] = config[model]['batch_size']
                self.model_settings[model]['learning_rate'] = config[model]['learning_rate']

    #---------------------------------------------------------------
    # Main processing function
    #---------------------------------------------------------------
    def analyze_flavor(self):

        # Loop through combinations of event type, jetR, and R_max
        self.AUC = {}
        for jet_pt_min in self.jet_pt_min_list:

            # Skip if no models are selected
            if not self.models:
                continue
        
            # Clear variables
            self.y = None
            self.X_particles = {}

            # Create output dir
            self.output_dir_i = os.path.join(self.output_dir, f'pt{jet_pt_min}')
            if not os.path.exists(self.output_dir_i):
                os.makedirs(self.output_dir_i)

            # Read input file into dataframe -- the files store the particle info as: (pt, eta, phi, pid)
            # Then transform these into a 3D numpy array (jets, particles, particle info)
            # The format of the particle info in X_particles will be: (pt, eta, phi, m, pid, charge)
            X_particles_total, self.y_total = self.load_training_data(jet_pt_min)

            # Determine total number of jets
            total_jets = int(self.y_total.size)
            total_jets_class2 = int(np.sum(self.y_total))
            total_jets_class1 = total_jets - total_jets_class2
            print(f'Total number of jets available: {total_jets_class1} ({self.class1_label}), {total_jets_class2} ({self.class2_label})')

            # If there is an imbalance, remove excess jets
            if self.balance_samples:
                if total_jets_class1 > total_jets_class2:
                    indices_to_remove = np.where( np.isclose(self.y_total,0) )[0][total_jets_class2:]
                elif total_jets_class1 < total_jets_class2:
                    indices_to_remove = np.where( np.isclose(self.y_total,1) )[0][total_jets_class1:]
                y_balanced = np.delete(self.y_total, indices_to_remove)
                X_particles_balanced = np.delete(X_particles_total, indices_to_remove, axis=0)
                total_jets = int(y_balanced.size)
                total_jets_class1 = int(np.sum(y_balanced))
                total_jets_class2 = total_jets - total_jets_class1
                print(f'Total number of jets available after balancing: {total_jets_class1} ({self.class1_label}), {total_jets_class2} ({self.class2_label})')
            else:
                y_balanced = self.y_total
                X_particles_balanced = X_particles_total

            # Reset the training,test,validation sizes based on the balanced number of jets
            self.n_total = total_jets
            self.n_train = int(self.n_total * self.train_frac)
            self.n_test = int(self.n_total * self.test_frac)
            self.n_val = self.n_total - self.n_train - self.n_test # int((self.n_total - self.n_test) * self.val_frac)
            print(f'n_train: {self.n_train}, n_test: {self.n_test}, n_val: {self.n_val}')

            # Shuffle dataset 
            idx = np.random.permutation(len(y_balanced))
            if y_balanced.shape[0] == idx.shape[0]:
                y_shuffled = y_balanced[idx]
                X_particles_shuffled = X_particles_balanced[idx]
            else:
                print(f'MISMATCH of shape: {y_shuffled.shape} vs. {idx.shape}')

            # Truncate the input arrays to the requested size
            self.y = y_shuffled[:self.n_total]
            self.X_particles_unfiltered = X_particles_shuffled[:self.n_total]
            print(f'y_shuffled sum: {np.sum(self.y)}')
            print(f'y_shuffled shape: {self.y.shape}')

            # Create additional sets of four-vectors in which a min-pt cut is applied -- the labels can stay the same
            if 'pfn' in self.models:
                for particle_pt_min in self.particle_pt_min_list:
                    self.X_particles[f'particle_pt_min{particle_pt_min}'] = filter_four_vectors(np.copy(self.X_particles_unfiltered), min_pt=particle_pt_min)

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
                X_EFP = self.X_particles['particle_pt_min0'][:,:,:4] # Remove pid,charge from self.X_particles
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
    # Load training data from set of input files into numpy arrays
    #---------------------------------------------------------------
    def load_training_data(self, jet_pt_min):

        for i,input_file in enumerate(self.input_files):
            print(f'Loading file {i+1}/{len(self.input_files)}...')

            jet_df = pd.read_csv(input_file, sep='\s+')
            X_particles, y, class_array = self.create_jet_array(jet_df, jet_pt_min)
            print(f'X_particles shape: {X_particles.shape}')
            print(f'y shape: {y.shape}')

            if i == 0:
                X_particles_total = X_particles
                y_total = y
                class_array_total = class_array
            else:

                # Zero pad according to the largest n_particles dimension
                n_particles_total = X_particles_total.shape[1]
                n_particles = X_particles.shape[1]
                if n_particles_total < n_particles:
                    X_particles_total = np.pad(X_particles_total, [(0,0), (0,n_particles-n_particles_total), (0,0)], mode='constant', constant_values=0)
                elif n_particles_total > n_particles:
                    X_particles = np.pad(X_particles, [(0,0), (0,n_particles_total-n_particles), (0,0)], mode='constant', constant_values=0)

                X_particles_total = np.concatenate([X_particles_total, X_particles])
                y_total = np.concatenate([y_total, y])
                class_array_total = np.concatenate([class_array_total, class_array])

            print()
            print(f'X_particles_total shape: {X_particles_total.shape}')
            print(f'y_total shape: {y_total.shape}')
            print()

            # If enough jets have been found, then return
            print(f'We have now found {y_total.shape[0]}/{self.n_total} training events.')
            if y_total.shape[0] > self.n_total:
                break

        print('Done loading!')
        print()

        # Plot statistics for each class
        classes, counts = np.unique(class_array_total, return_counts=True)
        classes_count_dict = dict(zip(classes, counts))
        print(f'class statistics: {classes_count_dict}') 
        print()
        plt.bar(list(classes_count_dict.keys()), classes_count_dict.values(), color='g', log=True)
        plt.ylabel("counts")
        plt.xlabel("class id")
        plt.savefig(os.path.join(self.output_dir, f'class_statistics_pt{jet_pt_min}.pdf'))
        plt.close()

        return X_particles_total, y_total

    #---------------------------------------------------------------
    # Parse the input file into a 3D array (jets, particles, particle info)
    # The particle info will be stored as: (pt, eta, phi, m, pid, charge)
    #---------------------------------------------------------------
    def create_jet_array(self, jet_df, jet_pt_min):

        # Set column name to get classes from
        if self.classification_type == 'jet':
            class_key = 'qg'
        elif self.classification_type == 'event':
            class_key = 'proc'

        # First, remove the particles outside the jets (only present in DIS events)
        jet_df = jet_df[jet_df.jet > 0]

        # Add columns of mass and charge
        jet_df['m'] = energyflow.pids2ms(jet_df['pid'], error_on_unknown=True)
        jet_df['charge'] = energyflow.pids2chrgs(jet_df['pid'], error_on_unknown=True)

        # Switch order: (pt, eta, phi, pid, m, charge) --> (pt, eta, phi, m, pid, charge)
        columns = list(jet_df.columns)
        index_pid = columns.index('pid')
        index_mass = columns.index('m')
        columns[index_pid], columns[index_mass] = columns[index_mass], columns[index_pid]
        jet_df = jet_df[columns]

        # Filter by jet pt
        jet_df = jet_df[jet_df['jetpT']>jet_pt_min]

        # Get statistics of each class type
        if self.classification_type == 'jet':
            class_array = jet_df[jet_df['ct']==1][class_key]
        elif self.classification_type == 'event':
            class_array = np.array([event[1][class_key].iloc[0] for event in jet_df.groupby('event')])

        #---
        # Set the ML labels based on the set of classes to classify

        # Remove entries that do not correspond to one of the requested class labels
        requested_classes = self.class1_ids + self.class2_ids
        labels_all = jet_df[class_key].to_numpy()
        mask = np.isin(labels_all, requested_classes)
        jet_df = jet_df[mask]

        # Get new list of class labels from masked dataframe
        if self.classification_type == 'jet':
            labels = jet_df[jet_df.ct==1][class_key].to_numpy()
        elif self.classification_type == 'event':
            labels = np.array([event[1][class_key].iloc[0] for event in jet_df.groupby('event')])

        # Find all labels from class1, and set them to 0
        labels_1 = np.invert(np.isin(labels, self.class1_ids)).astype(int)

        # Find all labels from class2, and set them to 1
        labels_2 = np.isin(labels, self.class2_ids).astype(int)

        # Check that the two are equal (i.e. that there are no unexpected class labels in the input file)
        if np.array_equal(labels_1, labels_2):
            labels = labels_1
        else:
            expected_ids = list(self.class_map.values())
            unique_ids = np.unique(labels)
            sys.exit(f'Unexpected class labels ({set(expected_ids).symmetric_difference(unique_ids)}) found in input file!')

        print(f'class1: {self.classes_class1} ({self.class1_ids}) (ML label 0), class2: {self.classes_class2} ({self.class2_ids}) (ML label 1)')
        print()
        #---

        # Check pdg values that are present
        pdg_values_present = np.unique(jet_df['pid'].values)
        print('pdg values in the particle list:')
        for pdg_value in pdg_values_present:
            print(f'  {pdg_value}: {Particle.from_pdgid(pdg_value)}')

        # Particles expected for c*tau > 1cm: (gamma, e-, mu-, pi+, K+, K_L0, K_S0, p+, n, Sigma+, Sigma-, Xi-, Xi0, Omega-, Lambda0)
        #     and antiparticles for (e-, mu-, pi+, K+, p+, n, Sigma+, Sigma-, Xi-, Xi0, Omega-, Lambda0)
        reference_particles_pdg = [22, 11, 13, 211, 321, 130, 310, 2212, 2112, 3222, 3112, 3312, 3322, 3334, 3122, 
                                    -11, -13, -211, -321, -2212, -2112, -3222, -3112, -3312, -3322, -3334, -3122]

        for pdg_value in reference_particles_pdg:
            if pdg_value not in pdg_values_present:
                print(f'WARNING: Missing particles: {Particle.from_pdgid(pdg_value)} not found in your accepted particles!')
        
        exit = False
        for pdg_value in pdg_values_present:
            if pdg_value not in reference_particles_pdg:
                print(f'WARNING: Extra particles: {Particle.from_pdgid(pdg_value)} was found in your accepted particles!')
        if exit:
            sys.exit()

        # Translate dataframe into 3D numpy array: (jets, particles, particle info)
        #                          where particle info is: (pt, eta, phi, m, pid, charge)
        # Based on: https://stackoverflow.com/questions/52621497/pandas-group-by-column-and-transform-the-data-to-numpy-array
        if self.classification_type == 'jet':

            # Process selection
            #   For classes = uds: always 99 for the LO DIS process
            #   For classes = qg: https://eic.github.io/software/pythia6.html
            if self.event_type == 'photoproduction':
                sys.exit(f'ERROR: process selection not yet implemented for jet classification for photoproduction events!')

            # First, drop unnecessary columns
            jet_df = jet_df.drop(columns=['proc', 'event', 'qg', 'ct', 'jetpT'])

            # Generate particle indices for each jet
            jet_df_grouped = jet_df.groupby(['jet'])
            particle_indices = jet_df_grouped.cumcount()

            # Zero pad                                                                                
            jet_df_zero_padded = jet_df.set_index(['jet', particle_indices]).unstack(fill_value=0).stack()

            # Group and convert to array
            jet_list = jet_df_zero_padded.groupby(level=0).apply(lambda x: x.values.tolist()).tolist()                                                                                                               
            jet_array = np.array(jet_list)
            print(f'(n_jets, n_particles, n_particle_info) = {jet_array.shape}')

        # For event classification, we stack all the jets in the event together (each jet with zero-padded block of fixed size)
        elif self.classification_type == 'event':

            # Create empty numpy array, which we will fill with zero-padded jets
            n_events = labels.size
            event_size = self.n_jets_max * self.n_particles_per_jet_max
            n_variables_per_particle = 6
            jet_array = np.zeros((n_events, event_size, n_variables_per_particle))
            print(f'(n_jets, n_particles, n_particle_info) = {jet_array.shape}')

            # Drop unnecessary columns
            jet_df = jet_df.drop(columns=['proc', 'qg', 'ct', 'jetpT'])

            # Group by event
            event_df_grouped = jet_df.groupby(['event'])
            
            # Loop through events and fill the numpy array
            event_index = 0
            for _,event_df in event_df_grouped:

                # Group by jet
                jet_df_grouped = event_df.groupby(['jet'])

                # Fill each jet into the numpy array
                jet_index = 0
                for _,jet_df in jet_df_grouped:

                    # Check that number of jets and number of particles are less than our allowed maximum
                    if jet_index >= self.n_jets_max:
                        sys.exit(f'ERROR: event {event_index} contains more than n_jets_max={self.n_jets_max} jets')
                    if len(jet_df.index) > self.n_particles_per_jet_max:
                        sys.exit(f'ERROR: event {event_index} jet {jet_index} contains more than n_particles_per_jet_max={self.n_particles_per_jet_max} particles')

                    # Convert dataframe to numpy array and add it to the main numpy array
                    jet_df = jet_df.drop(columns=['event', 'jet'])
                    jet_array_i = jet_df.to_numpy()

                    particle_index_min = jet_index*self.n_particles_per_jet_max
                    particle_index_max = particle_index_min + jet_array_i.shape[0]
                    jet_array[event_index, particle_index_min:particle_index_max, :] = jet_array_i

                    jet_index += 1
                
                event_index += 1

        return jet_array, labels, class_array

    #---------------------------------------------------------------
    # Compute some individual jet observables
    # TODO: speed up w/numba
    #---------------------------------------------------------------
    def compute_jet_observables(self):
        print('Compute jet observables...')
        print()

        self.qa_results = defaultdict(list)
        for particle_pt_min in self.particle_pt_min_list:
            self.qa_observables = [f'jet_charge_ptmin{particle_pt_min}_k{kappa}' for kappa in self.kappa]
            self.qa_observables += [f'jet_charge0_ptmin{particle_pt_min}_k{kappa}_multiplicity' for kappa in self.kappa]
            self.qa_observables += [f'particle_multiplicity_ptmin{particle_pt_min}']
            self.qa_observables += [f'strange_tagger_ptmin{particle_pt_min}']

        # Compute jet charge for each jet collection
        print('  Computing jet charge...')
        for particle_pt_min in self.particle_pt_min_list:
            print(f'    for jets with particle_pt_min={particle_pt_min}')
            for kappa in self.kappa:
                print(f'      kappa={kappa}...')

                charge0_jets_with_charged_constituents = {}

                for i,jet in enumerate(self.X_particles[f'particle_pt_min{particle_pt_min}']):

                    jet_charge = 0
                    jet_pt = 0
                    for particle in jet:
                        pt = particle[0]
                        charge = particle[5]
                        jet_pt += pt
                        jet_charge += charge * np.power(pt, kappa)
                    jet_charge = jet_charge / np.power(jet_pt, kappa)
                    self.qa_results[f'jet_charge_ptmin{particle_pt_min}_k{kappa}'].append(jet_charge)

                    # Check properties of charge=0 jets
                    if np.isclose(jet_charge, 0.):
                        pid = jet[:,4]
                        pid_nonzero = pid[ pid != 0]
                        self.qa_results[f'jet_charge0_ptmin{particle_pt_min}_k{kappa}_multiplicity'].append(pid_nonzero.size)

                        # Check that the jet contains no charged particles
                        neutral_pids = np.array([22, 130, 310, 2112, 3322, 3122,
                                                -2112, -3322, -3122])
                        charged_pids = np.array([11, 13, 211, 321, 2212, 3222, 3112, 3312, 3334
                                                -11, -13, -211, -321, -2212, -3222, -3112, -3312, -3334])
                        if np.any(np.in1d(pid_nonzero, charged_pids)):
                            charge0_jets_with_charged_constituents[i] = jet
                            print(f'WARNING: unexpected jet charge={jet_charge}')
                            print(f'pid: {jet[:,4][ jet[:,4] != 0]}')
                            print(f'pid: {jet[:pid_nonzero.size,4]}')
                            print(f'charge: {jet[:pid_nonzero.size,5]}')
                            print(f'pt: {jet[:,0][ jet[:,0] != 0]}')
                            print(f'eta: {jet[:,1][ jet[:,1] != 0]}')
                            print(f'phi: {jet[:,2][ jet[:,2] != 0]}')
                            print()

                if charge0_jets_with_charged_constituents:        
                    print(f'WARNING: {len(charge0_jets_with_charged_constituents.keys())} jets with charge=0 (kappa={kappa}) have charged constituents!')
                    print()

        print('  Done.')
        print()

        # Compute some other jet observables
        print('  Computing additional observables...')
        for particle_pt_min in self.particle_pt_min_list:
            print(f'    for jets with particle_pt_min={particle_pt_min}')
            for jet in self.X_particles[f'particle_pt_min{particle_pt_min}']:

                pid = jet[:,4]
                pid_nonzero = pid[ pid != 0]

                # Compute particle multiplicity
                self.qa_results[f'particle_multiplicity_ptmin{particle_pt_min}'].append(pid_nonzero.size)

                # Compute whether jet has a strange hadron in it
                strange_particle_pdg = [321, 130, 310, 3222, 3112, 3312, 3322, 3334, 3122, 
                                        -321, -3222, -3112, -3312, -3322, -3334, -3122]
                found_strange_hadron = np.any(np.in1d(pid_nonzero, strange_particle_pdg))
                self.qa_results[f'strange_tagger_ptmin{particle_pt_min}'].append(found_strange_hadron)

        print('Done.')

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
            if 'efp' in model:
                for d in range(1, self.dmax+1):
                    if model == 'efp_linear':
                        self.fit_efp_linear(model, model_settings, d)
                    if model == 'efp_dnn':
                        self.fit_efp_dnn(model, model_settings, d)
                if model == 'efp_lasso':
                    self.fit_efp_lasso(model, model_settings, self.d_lasso)

            # Deep sets
            if model == 'pfn':

                for particle_pt_min in self.particle_pt_min_list:

                    if model_settings['pid']:
                        model_label = f'pfn_pid_minpt{particle_pt_min}'
                        self.fit_pfn(model_label, model_settings, self.y, self.X_particles[f'particle_pt_min{particle_pt_min}'], pid=True)

                    if model_settings['nopid']:
                        model_label = f'pfn_nopid_minpt{particle_pt_min}'
                        self.fit_pfn(model_label, model_settings, self.y, self.X_particles[f'particle_pt_min{particle_pt_min}'], pid=False, charge=False)

                    if model_settings['charge']:
                        model_label = f'pfn_charge_minpt{particle_pt_min}'
                        self.fit_pfn(model_label, model_settings, self.y, self.X_particles[f'particle_pt_min{particle_pt_min}'], pid=False, charge=True)
                
            if model == 'efn':
                self.fit_efn(model, model_settings)

        # Plot traditional observables
        for observable in self.qa_observables:
            if self.y.size == len(self.qa_results[observable]):
                self.roc_curve_dict_lasso[observable] = sklearn.metrics.roc_curve(self.y, -np.array(self.qa_results[observable]).astype(np.float))
                self.roc_curve_dict[observable] = sklearn.metrics.roc_curve(self.y, -np.array(self.qa_results[observable]).astype(np.float))
            else:
                print(f'Skip constructing ROC curve for observable={observable}, due to mismatch with number of labels')

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
    def fit_pfn(self, model, model_settings, y, X_particles, pid=False, charge=False):
    
        # Convert labels to categorical
        Y_PFN = energyflow.utils.to_categorical(y, num_classes=2)
                        
        # (pT,y,phi,pid/charge)
        if charge:
            X_PFN = X_particles[:,:,[0,1,2,5]]
        else:
            X_PFN = X_particles[:,:,[0,1,2,4]]

        # Preprocess by centering jets and normalizing pts
        for x_PFN in X_PFN:
            mask = x_PFN[:,0] > 0
            yphi_avg = np.average(x_PFN[mask,1:3], weights=x_PFN[mask,0], axis=0)
            x_PFN[mask,1:3] -= yphi_avg
            x_PFN[mask,0] /= x_PFN[:,0].sum()
        
        # Handle particle id channel
        if pid:
            self.my_remap_pids(X_PFN)
        else:
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
        X_EFN = self.X_particles['particle_pt_min0'][:,:,:4] # Remove pid,charge from self.X_particles
        
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
        # PDGid to small float dictionary (neutral & positive charge are assigned a positive number)
        PID2FLOAT_MAP = { 0: +0.0,       # no particle
                          22: +0.05,     # gamma
                          11: -0.1,      # e^-
                         -11: +0.1,      # e^+
                          13: -0.2,      # mu^-
                         -13: +0.2,      # mu^+
                          211: +0.3,     # pi^+
                         -211: -0.3,     # pi^-
                          321: +0.4,     # K^+
                         -321: -0.4,     # K^-
                          130: +0.5,     # K_L^0
                          310: +0.6,     # K_S^0
                          2212: +0.7,    # p
                         -2212: -0.7,    # p bar
                          2112: +0.8,    # n
                         -2112: +0.9,    # n bar
                          3222: +1.0,    # Sigma^+ (uus)
                         -3222: -1.0,    # Sigma^+ bar
                          3112: -1.1,    # Sigma^- (dds)
                         -3112: +1.1,    # Sigma^- bar
                          3312: -1.2,    # Xi^- (dss)
                         -3312: +1.2,    # Xi^- bar
                          3322: +1.3,    # Xi^0 (uss)
                         -3322: +1.4,    # Xi^0 bar
                          3334: -1.5,    # Omega^- (sss)
                         -3334: +1.5,    # Omega^- bar
                          3122: +1.6,    # Lambda^0 (uds)
                         -3122: +1.7}    # Lambda^0 bar

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

            # If same number of entries as labels, then plot separated by class label 
            if self.y.shape[0] == len(result) and 'multiplicity' not in qa_observable:

                class1_indices = 1 - self.y
                class2_indices = self.y
                result_class1 = result[class1_indices.astype(bool)]
                result_class2 = result[class2_indices.astype(bool)]

                # Set some labels
                if 'jet_charge' in qa_observable:
                    xlabel = rf'$Q_{{\kappa}}$'
                    ylabel = rf'$\frac{{1}}{{\sigma}} \frac{{d\sigma}}{{dQ_{{\kappa}} }}$'
                    bins = np.linspace(-2, 2., 100)
                else:
                    ylabel = ''
                    xlabel = rf'{qa_observable}'
                    bins = np.linspace(0, np.amax(result_class2), 20)
                plt.xlabel(xlabel, fontsize=14)
                plt.ylabel(ylabel, fontsize=16)

                if qa_observable == 'jet_pt':
                    stat='count'
                else:
                    stat='density'

                # Construct dataframes for histplot
                df_class1 = pd.DataFrame(result_class1, columns=[xlabel])
                df_class2 = pd.DataFrame(result_class2, columns=[xlabel])
                
                # Add label columns to each df to differentiate them for plotting
                df_class1['generator'] = np.repeat(self.class1_label, result_class1.shape[0])
                df_class2['generator'] = np.repeat(self.class2_label, result_class2.shape[0])
                df = df_class1.append(df_class2, ignore_index=True)

                # Histplot
                h = sns.histplot(df, x=xlabel, hue='generator', stat=stat, bins=bins, element='step', common_norm=False)
                h.legend_.set_title(None)
                plt.setp(h.get_legend().get_texts(), fontsize='14') # for legend text

                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir_i, f'{qa_observable}.pdf'))
                plt.close()

            # Otherwise, plot distribution without dividing by class label
            else:

                if 'multiplicity' in qa_observable:
                    bins = np.linspace(-0.5, 30.5, 32)
                else:
                    bins = 'auto'
                plt.hist(result, bins=bins)

                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir_i, f'{qa_observable}.pdf'))
                plt.close()

    #---------------------------------------------------------------
    # Plot q vs. g
    #---------------------------------------------------------------
    def plot_observable(self, X, y_train, xlabel='', ylabel='', filename='', xfontsize=12, yfontsize=16, logx=False, logy=False):
            
        class1_indices = 1 - y_train
        class2_indices = y_train

        observable_class1 = X[class1_indices.astype(bool)]
        observable_class2 = X[class2_indices.astype(bool)]

        df_class1 = pd.DataFrame(observable_class1, columns=[xlabel])
        df_class2 = pd.DataFrame(observable_class2, columns=[xlabel])

        df_class1['generator'] = np.repeat(self.class1_label, observable_class1.shape[0])
        df_class2['generator'] = np.repeat(self.class2_label, observable_class2.shape[0])
        df = df_class1.append(df_class2, ignore_index=True)

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

        class1_indices = 1 - self.y
        class2_indices = self.y
        X_q = X_EFP_d[class1_indices.astype(bool)]
        X_g = X_EFP_d[class2_indices.astype(bool)]

        # Get labels
        graphs = [str(x) for x in self.graphs[:4]]

        # Construct dataframes for scatter matrix plotting
        df_class1 = pd.DataFrame(X_class1, columns=graphs)
        df_class1 = pd.DataFrame(X_class2, columns=graphs)
        
        # Add label columns to each df to differentiate them for plotting
        df_class1['generator'] = np.repeat(self.class1_label, X_class1.shape[0])
        df_class2['generator'] = np.repeat(self.class2_label, X_class2.shape[0])
        df = df_class1.append(df_class2, ignore_index=True)

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
                        default='config/u_d.yaml',
                        help='Path of config file for analysis')
    parser.add_argument('-o', '--outputDir', action='store',
                        type=str, metavar='outputDir',
                        default='./TestOutput',
                        help='Output directory for output to be written to')

    # Parse the arguments
    args = parser.parse_args()

    print('Configuring...')
    print('configFile: \'{0}\''.format(args.configFile))
    print('ouputDir: \'{0}\''.format(args.outputDir))

    # If invalid configFile is given, exit
    if not os.path.exists(args.configFile):
        print('File \"{0}\" does not exist! Exiting!'.format(args.configFile))
        sys.exit(0)

    analysis = AnalyzeFlavor(config_file=args.configFile, output_dir=args.outputDir)
    analysis.analyze_flavor()