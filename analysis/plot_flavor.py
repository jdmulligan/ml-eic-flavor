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

        if 'pfn' in self.models and 'efn' in self.models:
            roc_list = {}
            roc_list['PFN'] = self.roc_curve_dict['pfn']
            roc_list['EFN'] = self.roc_curve_dict['efn']
            self.plot_roc_curves(roc_list, jet_pt_min)

        if 'efp_linear' in self.models:
             roc_list = {}
             for d in range(3, self.dmax+1):
                 roc_list[f'EFP (d = {d}), Linear'] = self.roc_curve_dict['efp_linear'][d]
             self.plot_roc_curves(roc_list, jet_pt_min)

        if 'efp_dnn' in self.models:
             roc_list = {}
             for d in range(3, self.dmax+1):
                 roc_list[f'EFP (d = {d}), DNN'] = self.roc_curve_dict['efp_dnn'][d]
             self.plot_roc_curves(roc_list, jet_pt_min)

        if 'efp_linear' in self.models and 'efp_lasso' in self.models:
            roc_list = {}
            roc_list[f'EFP (d = {self.d_lasso_efp}), Linear'] = self.roc_curve_dict['efp_linear'][self.d_lasso_efp]
            roc_list['thrust'] = self.roc_curve_dict_lasso['thrust']
            roc_list['jet_angularity'] = self.roc_curve_dict_lasso['jet_angularity']
            roc_list['jet_theta_g'] = self.roc_curve_dict_lasso['jet_theta_g']
            roc_list['zg'] = self.roc_curve_dict_lasso['zg']
            for alpha in self.efp_alpha_list:
                roc_list[rf'Lasso $(\alpha = {alpha})$, EFP'] = self.roc_curve_dict_lasso['efp_lasso'][alpha]
            self.plot_roc_curves(roc_list, jet_pt_min)
            self.plot_significance_improvement(roc_list, jet_pt_min)

    #--------------------------------------------------------------- 
    # Plot ROC curves
    #--------------------------------------------------------------- 
    def plot_roc_curves(self, roc_list, jet_pt_min):
    
        plt.plot([0, 1], [0, 1], 'k--') # dashed diagonal
        plt.axis([0, 1, 0, 1])
        plt.title('JEWEL vs. PYTHIA8     ' + r'$100<p_{\mathrm{T,jet}}<125\;\mathrm{GeV}$', fontsize=14)
        if self.plot_title:
            plt.title(rf'$p_T min = {jet_pt_min}$', fontsize=14)
        plt.xlabel('False AA Rate', fontsize=16)
        plt.ylabel('True AA Rate', fontsize=16)
        #plt.ylabel(rf'$\varepsilon_{{\rm{{true positive}} }}^{{\rm{{AA}} }}$', fontsize=16)
        plt.grid(True)
    
        for label,value in roc_list.items():
            index=0
            if label in ['PFN', 'EFN', 'jet_mass', 'jet_angularity', 'LHA', 'thrust', 'pTD', 'hadron_z', 'zg', 'jet_theta_g'] or 'multiplicity' in label or 'PFN' in label or 'EFN' in label or 'pfn' in label:
                linewidth = 4
                alpha = 0.5
                linestyle = self.linestyle(label)
                color=self.color(label)
                legend_fontsize = 12
                if label == 'jet_mass':
                    label = r'$m_{\mathrm{jet}}$'
                if label == 'jet_angularity':
                    label = r'$\lambda_1$ (girth)'
                    linewidth = 2
                    alpha = 0.6
                if label == 'thrust':
                    label = r'$\lambda_2$ (thrust)'
                    linewidth = 2
                    alpha = 0.6
                if label == 'jet_theta_g':
                    label = r'$\theta_{\mathrm{g}}$'
                    linewidth = 2
                    alpha = 0.6
                if label == 'zg':
                    label = r'$z_{\mathrm{g}}$'
                    linewidth = 2
                    alpha = 0.6
                if label == 'PFN':
                    label = 'Particle Flow Network'
                if label == 'EFN':
                    label = 'Energy Flow Network'
                if label == 'PFN_hard':
                    label = 'Jet'
                if label == 'PFN_hard_min_pt':
                    label = rf'Jet, $p_{{ \rm{{T}} }}^{{ \rm{{particle}} }} > 1$ GeV'
                    alpha = 0.6
                    linewidth = 2
                if label == 'PFN_background':
                    label = 'Jet + Background'
                if label == 'PFN_background_min_pt':
                    label = rf'Jet + Background, $p_{{ \rm{{T}} }}^{{ \rm{{particle}} }} > 1$ GeV'
                    alpha = 0.6
                    linewidth = 2
                    legend_fontsize = 11
                if label == 'pfn_hard':
                    label = 'Jet'
                if label == 'pfn_beforeCS':
                    label = 'Jet + Background (before subtraction)'   
                    alpha = 0.6
                    legend_fontsize = 11
                    linewidth = 3             
                if label == 'pfn_afterCS':
                    label = rf'Jet + Background ($R_{{\rm{{max}}}}=0.25$)' 
                if label == 'pfn_afterCS_Rmax1':
                    label = rf'Jet + Background ($R_{{\rm{{max}}}}=1.0$)'
            elif 'Lasso' in label:
                linewidth = 2
                alpha = 1
                linestyle = 'solid'
                color=self.color(label)
                legend_fontsize = 10
                reg_param = float(re.search('= (.*)\)', label).group(1))
                if 'Nsub' in label:
                    n_terms = self.N_terms_lasso['nsub_lasso'][reg_param]
                    print(self.observable_lasso['nsub_lasso'][reg_param])
                elif 'EFP' in label:
                    n_terms = self.N_terms_lasso['efp_lasso'][reg_param]

                if 'Nsub' in label:
                    label = rf'$\prod_{{N,\beta}} \left( \tau_N^{{\beta}} \right) ^{{c_{{N,\beta}} }}$'
                elif 'EFP' in label:
                    label = rf'$\sum_{{G}} c_{{G}} \rm{{EFP}}_{{G}}$'
                label += f', {n_terms} terms'

            elif 'DNN' in label or 'EFP' in label or 'nsub' in label:
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
    # Plot Significance improvement
    #--------------------------------------------------------------- 
    def plot_significance_improvement(self, roc_list, jet_pt_min):
    
        plt.axis([0, 1, 0, 3])
        plt.title('JEWEL vs. PYTHIA8     ' + r'$100<p_{\mathrm{T,jet}}<125\;\mathrm{GeV}$', fontsize=14)
        if self.plot_title:
            plt.title(rf'$p_T min = {jet_pt_min}$', fontsize=14)
        plt.xlabel('True AA Rate', fontsize=16)
        plt.ylabel('Significance improvement', fontsize=16)
        plt.grid(True)
            
        for label,value in roc_list.items():
            index=0
            if label in ['PFN', 'EFN', 'jet_mass', 'jet_angularity', 'LHA', 'thrust', 'pTD', 'hadron_z', 'zg', 'jet_theta_g'] or 'multiplicity' in label or 'PFN' in label or 'EFN' in label or 'pfn' in label:
                linewidth = 4
                alpha = 0.5
                linestyle = self.linestyle(label)
                color=self.color(label)
                legend_fontsize = 12
                if label == 'jet_angularity':
                    label = r'$\lambda_1$ (girth)'
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
                if 'Nsub' in label:
                    n_terms = self.N_terms_lasso['nsub_lasso'][reg_param]
                    print(self.observable_lasso['nsub_lasso'][reg_param])
                elif 'EFP' in label:
                    n_terms = self.N_terms_lasso['efp_lasso'][reg_param]

                if 'Nsub' in label:
                    label = rf'$\prod_{{N,\beta}} \left( \tau_N^{{\beta}} \right) ^{{c_{{N,\beta}} }}$'
                elif 'EFP' in label:
                    label = rf'$\sum_{{G}} c_{{G}} \rm{{EFP}}_{{G}}$'
                label += f', {n_terms} terms'

            elif 'DNN' in label or 'EFP' in label or 'nsub' in label:
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
            plt.plot(TPR, TPR/np.sqrt(FPR+0.001), linewidth=linewidth, label=label,
                     linestyle=linestyle, alpha=alpha, color=color)
         
        plt.legend(loc='best', fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir_i, f'Significance_improvement_{self.significance_plot_index}.pdf'))
        plt.close()

        self.significance_plot_index += 1

    #---------------------------------------------------------------
    # Plot AUC as a function of K
    #---------------------------------------------------------------
    def plot_AUC_convergence(self, output_dir, auc_list, event_type, jetR, jet_pt_bin, R_max):
    
        plt.axis([0, self.K_list[-1], 0, 1])
        if self.plot_title:
            plt.title(rf'{event_type} event: $R = {jetR}, p_T = {jet_pt_bin}, R_{{max}} = {R_max}$', fontsize=14)
        plt.xlabel('K', fontsize=16)
        plt.ylabel('AUC', fontsize=16)

        for label,value in auc_list.items():
        
            if 'hard' in label:
                color = sns.xkcd_rgb['dark sky blue']
                label_suffix = ' (no background)'
            if 'combined' in label:
                color = sns.xkcd_rgb['medium green']
                label_suffix = ' (thermal background)'

            if 'pfn' in label:
                AUC_PFN = value[0]
                label = f'PFN{label_suffix}'
                plt.axline((0, AUC_PFN), (1, AUC_PFN), linewidth=4, label=label,
                           linestyle='solid', alpha=0.5, color=color)
            elif 'efn' in label:
                AUC_EFN = value[0]
                label = f'EFN{label_suffix}'
                plt.axline((0, AUC_EFN), (1, AUC_EFN), linewidth=4, label=label,
                           linestyle='solid', alpha=0.5, color=color)
            elif 'neural_network' in label:
                label = f'DNN{label_suffix}'
                plt.plot(self.K_list, value, linewidth=2,
                         linestyle='solid', alpha=0.9, color=color,
                         label=label)
                    
        plt.legend(loc='lower right', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'AUC_convergence{self.auc_plot_index}.pdf'))
        plt.close()

        self.auc_plot_index += 1

    #---------------------------------------------------------------
    # Get color for a given label
    #---------------------------------------------------------------
    def color(self, label):

        color = None
        if label in ['PFN', 'PFN_hard', 'EFP (d = 7), Linear']:
            color = sns.xkcd_rgb['faded purple'] 
        elif label in ['EFN', 'EFN_hard']:
            color = sns.xkcd_rgb['faded red']    
        elif label in ['EFP (d = 6), Linear']:
            color = sns.xkcd_rgb['dark sky blue']    
        elif label in ['pfn_afterCS_Rmax1']:
            color = sns.xkcd_rgb['light lavendar']    
        elif label in ['EFN_background', 'EFP (d = 5), Linear']:
            color = sns.xkcd_rgb['medium green']  
        elif label in ['EFP (d = 3), Linear', rf'Lasso $(\alpha = {self.efp_alpha_list[1]})$, EFP']:
            color = sns.xkcd_rgb['watermelon'] 
        elif label in ['EFP (d = 4), Linear', rf'Lasso $(\alpha = {self.efp_alpha_list[0]})$, EFP']:
            color = sns.xkcd_rgb['light brown'] 
        elif label in ['jet_theta_g']:
            color = sns.xkcd_rgb['medium brown']
        else:
            color = sns.xkcd_rgb['almost black']
        #  'pfn_beforeCS', 'pfn_beforeCS_min_pt', 

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