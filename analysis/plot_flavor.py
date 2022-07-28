#!/usr/bin/env python3

"""
Plot pp vs. AA classification performance
"""

import os
import sys
import argparse
import yaml
import re
import pickle

# Data analysis and plotting
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import seaborn as sns
sns.set_context('paper', rc={'font.size':18,'axes.titlesize':18,'axes.labelsize':18})

# Base class
sys.path.append('.')
from base import common_base

################################################################
class PlotFlavor(common_base.CommonBase):

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

        # Suffix for plot outputfile names
        self.roc_plot_index = 0
        self.significance_plot_index = 0
        self.auc_plot_index = 0

        self.plot_title = False
                
    #---------------------------------------------------------------
    # Initialize config file into class members
    #---------------------------------------------------------------
    def initialize_config(self):
    
        # Read config file
        with open(self.config_file, 'r') as stream:
          config = yaml.safe_load(stream)
          
        self.jet_pt_min_list = config['jet_pt_min_list']
        self.kappa = config['kappa']

        self.flavor_type = config['flavor_type']
        class_labels = self.flavor_type.split('__')
        self.class1_label = class_labels[0]
        self.class2_label = class_labels[1]

        self.models = config['models']
        self.dmax = config['dmax']
        self.efp_measure = config['efp_measure']
        self.efp_beta = config['efp_beta']
        
        # Initialize model-specific settings
        self.efp_alpha_list = config['efp_lasso']['alpha']
        self.d_lasso_efp = config['efp_lasso']['d_lasso']

    #---------------------------------------------------------------
    # Main processing function
    #---------------------------------------------------------------
    def plot_flavor(self):
    
        # Loop through min jet pt datasets
        for jet_pt_min in self.jet_pt_min_list:

            if not self.models:
                continue
        
            # Create output dir
            self.output_dir_i = os.path.join(self.output_dir, f'pt{jet_pt_min}')
            if not os.path.exists(self.output_dir_i):
                os.makedirs(self.output_dir_i)

            # Load ML results from file
            self.key_suffix = f'pt{jet_pt_min}'
            roc_filename = os.path.join(self.output_dir_i, f'ROC{self.key_suffix}.pkl')
            with open(roc_filename, 'rb') as f:
                self.roc_curve_dict = pickle.load(f)
                self.AUC = pickle.load(f)

            lasso_filename = os.path.join(self.output_dir_i, f'ROC{self.key_suffix}_lasso.pkl')
            if os.path.exists(lasso_filename):
                with open(lasso_filename, 'rb') as f_lasso:
                    self.roc_curve_dict_lasso = pickle.load(f_lasso)
                    self.N_terms_lasso = pickle.load(f_lasso)
                    self.observable_lasso = pickle.load(f_lasso)

            # Plot models for a single setting
            self.plot_models(jet_pt_min)

    #---------------------------------------------------------------
    # Plot several versions of ROC curves and significance improvement
    #---------------------------------------------------------------
    def plot_models(self, jet_pt_min):

        if 'pfn' in self.models:
            roc_list = {}
            roc_list['PFN'] = self.roc_curve_dict['pfn']
            for kappa in self.kappa:
                roc_list[f'jet_charge_k{kappa}'] = self.roc_curve_dict[f'jet_charge_k{kappa}']
            self.plot_roc_curves(roc_list, jet_pt_min)

        if 'pfn' in self.models and 'efn' in self.models:
            roc_list = {}
            roc_list['PFN'] = self.roc_curve_dict['pfn']
            roc_list['EFN'] = self.roc_curve_dict['efn']
            self.plot_roc_curves(roc_list, jet_pt_min)

        if 'efp_linear' in self.models:
             roc_list = {}
             for d in range(3, self.dmax+1):
                 roc_list[f'EFP (d = {d}), Linear'] = self.roc_curve_dict['efp_linear'][d]
             for kappa in self.kappa:
                 roc_list[f'jet_charge_k{kappa}'] = self.roc_curve_dict[f'jet_charge_k{kappa}']
             self.plot_roc_curves(roc_list, jet_pt_min)

        if 'efp_dnn' in self.models:
             roc_list = {}
             for d in range(3, self.dmax+1):
                 roc_list[f'EFP (d = {d}), DNN'] = self.roc_curve_dict['efp_dnn'][d]
             self.plot_roc_curves(roc_list, jet_pt_min)

        if 'efp_linear' in self.models and 'efp_lasso' in self.models:
            roc_list = {}
            roc_list[f'EFP (d = {self.d_lasso_efp}), Linear'] = self.roc_curve_dict['efp_linear'][self.d_lasso_efp]
            for alpha in self.efp_alpha_list:
                roc_list[rf'Lasso $(\alpha = {alpha})$, EFP'] = self.roc_curve_dict_lasso['efp_lasso'][alpha]
            for kappa in self.kappa:
                roc_list[f'jet_charge_k{kappa}'] = self.roc_curve_dict[f'jet_charge_k{kappa}']
            self.plot_roc_curves(roc_list, jet_pt_min)

    #--------------------------------------------------------------- 
    # Plot ROC curves
    #--------------------------------------------------------------- 
    def plot_roc_curves(self, roc_list, jet_pt_min):
    
        plt.plot([0, 1], [0, 1], 'k--') # dashed diagonal
        plt.axis([0, 1, 0, 1])
        plt.title(f'{self.class1_label} vs. {self.class2_label}     ' + rf'$p_{{\mathrm{{T,jet}}}}>{jet_pt_min}\;\mathrm{{GeV}}$', fontsize=14)
        if self.plot_title:
            plt.title(rf'$p_T min = {jet_pt_min}$', fontsize=14)
        plt.xlabel(f'False {self.class1_label} Rate', fontsize=16)
        plt.ylabel(f'True {self.class1_label} Rate', fontsize=16)
        plt.grid(True)
    
        for label,value in roc_list.items():
            index=0
            if label in ['PFN', 'EFN', 'pfn'] or 'jet_charge' in label:
                linewidth = 4
                alpha = 0.5
                linestyle = self.linestyle(label)
                color=self.color(label)
                legend_fontsize = 12
                if 'jet_charge' in label:
                    label = label
                    linewidth = 2
                    alpha = 0.6
                if label == 'PFN':
                    label = 'Particle Flow Network'
                if label == 'EFN':
                    label = 'Energy Flow Network'
            elif 'Lasso' in label:
                linewidth = 2
                alpha = 1
                linestyle = 'solid'
                color=self.color(label)
                legend_fontsize = 10
                reg_param = float(re.search('= (.*)\)', label).group(1))
                if 'EFP' in label:
                    n_terms = self.N_terms_lasso['efp_lasso'][reg_param]
                    label = rf'$\sum_{{G}} c_{{G}} \rm{{EFP}}_{{G}}$'
                label += f', {n_terms} terms'

            elif 'DNN' in label or 'EFP' in label:
                linewidth = 2
                alpha = 0.9
                linestyle = 'solid'
                color=self.color(label)
                legend_fontsize = 12
            else:
                linewidth = 2
                linestyle = 'solid'
                alpha = 0.9
                color = sns.xkcd_rgb['almost black']
                legend_fontsize = 12
  
            FPR = value[0]
            TPR = value[1]
            plt.plot(FPR, TPR, linewidth=linewidth, label=label,
                     linestyle=linestyle, alpha=alpha, color=color)
                    
        plt.legend(loc='lower right', fontsize=legend_fontsize)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir_i, f'ROC_{self.roc_plot_index}.pdf'))
        plt.close()

        self.roc_plot_index += 1

    #---------------------------------------------------------------
    # Get color for a given label
    #---------------------------------------------------------------
    def color(self, label):

        color = None
        if label in ['PFN', 'PFN_hard', 'EFP (d = 7), Linear', 'EFP (d = 7), DNN']:
            color = sns.xkcd_rgb['faded purple'] 
        elif label in ['EFN', 'EFN_hard', 'jet_charge_k0']:
            color = sns.xkcd_rgb['faded red']    
        elif label in ['EFP (d = 6), Linear', 'EFP (d = 6), DNN']:
            color = sns.xkcd_rgb['dark sky blue']    
        elif label in ['jet_charge_k0']:
            color = sns.xkcd_rgb['light lavendar']    
        elif label in ['EFN_background', 'EFP (d = 5), Linear', 'EFP (d = 5), DNN']:
            color = sns.xkcd_rgb['medium green']  
        elif label in ['EFP (d = 3), Linear', 'EFP (d = 3), DNN', rf'Lasso $(\alpha = {self.efp_alpha_list[1]})$, EFP']:
            color = sns.xkcd_rgb['watermelon'] 
        elif label in ['EFP (d = 4), Linear', 'EFP (d = 4), DNN', rf'Lasso $(\alpha = {self.efp_alpha_list[0]})$, EFP']:
            color = sns.xkcd_rgb['light brown'] 
        elif label in ['jet_charge_k0.3']:
            color = sns.xkcd_rgb['medium brown']
        else:
            color = sns.xkcd_rgb['almost black']

        return color

    #---------------------------------------------------------------
    # Get linestyle for a given label
    #---------------------------------------------------------------
    def linestyle(self, label):
 
        linestyle = None
        if 'PFN' in label and 'min_pt' in label:
            linestyle = 'dotted'
        elif 'PFN' in label or 'EFN' in label or 'DNN' in label or 'pfn' in label:
            linestyle = 'solid'
        else:
            linestyle = 'dotted'

        return linestyle
            
##################################################################
if __name__ == '__main__':

    # Define arguments
    parser = argparse.ArgumentParser(description='Plot ROC curves')
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

    analysis = PlotFlavor(config_file=args.configFile, output_dir=args.outputDir)
    analysis.plot_flavor()