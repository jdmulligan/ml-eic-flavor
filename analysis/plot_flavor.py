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
        self.roc_plot_index = {type:0 for type in self.reference_particle_input_types}

        self.plot_title = True
                
    #---------------------------------------------------------------
    # Initialize config file into class members
    #---------------------------------------------------------------
    def initialize_config(self):
    
        # Read config file
        with open(self.config_file, 'r') as stream:
          config = yaml.safe_load(stream)
          
        self.event_type = config['event_type']
        if self.event_type == 'photoproduction':
            self.leading_jet_pt_min = config['leading_jet_pt_min']
            self.subleading_jet_pt_min = config['subleading_jet_pt_min']
            self.jet_pt_min_list = [self.leading_jet_pt_min]
        elif self.event_type == 'dis':
            self.jet_pt_min_list = config['jet_pt_min_list']

        self.kappa = config['kappa']

        if self.event_type == 'photoproduction':
            self.class1_label = r'$qq/q\bar{q}$'
            self.class2_label = r'$gg$'
        elif self.event_type == 'dis':
            self.classes = config['classes']
            class_labels = self.classes.split('__')
            self.class1_label = class_labels[0]
            self.class2_label = class_labels[1]

        self.particle_input_type_list = config['particle_input']
        if self.event_type == 'photoproduction':
            self.reference_particle_input_types = ['leading', 'leading+subleading', 'all']
        elif self.event_type == 'dis':
            self.reference_particle_input_types = ['in', 'out', 'in+out']

        self.particle_pt_min_list = config['particle_pt_min_list']

        self.models = config['models']
        
    #---------------------------------------------------------------
    # Main processing function
    #---------------------------------------------------------------
    def plot_flavor(self):
    
        # Loop through combinations of jet_pt_min, particle_input type
        for jet_pt_min in self.jet_pt_min_list:

            # Load ROC curves for all possible particle_input_type
            self.output_dir_dict = {}
            self.roc_curve_dict = {}
            self.AUC_dict = {}
            for particle_input_type in self.reference_particle_input_types:
            
                # Load ML results from file
                self.output_dir_dict[particle_input_type] = os.path.join(self.output_dir, f'pt{jet_pt_min}_{particle_input_type}')
                roc_filename = os.path.join(self.output_dir_dict[particle_input_type], f'ROCpt{jet_pt_min}_{particle_input_type}.pkl')
                if os.path.exists(roc_filename):
                    with open(roc_filename, 'rb') as f:
                        self.roc_curve_dict[particle_input_type] = pickle.load(f)
                        self.AUC_dict[particle_input_type] = pickle.load(f)
            print(f'We found output directories for the following: {list(self.roc_curve_dict.keys())}')

            # Plot models
            self.plot_models(jet_pt_min)

    #---------------------------------------------------------------
    # Plot several versions of ROC curves
    #---------------------------------------------------------------
    def plot_models(self, jet_pt_min):

        #--------------------------
        # First, make plots for each particle_input_type specified in config file

        for particle_input_type in self.particle_input_type_list:

            # Plot pid vs. charge vs. nopid for each particle_pt_min
            type = 'fixed_ptmin'
            for particle_pt_min in self.particle_pt_min_list:

                pfn_charge_label = f'pfn_charge_minpt{particle_pt_min}'
                pfn_pid_label = f'pfn_pid_minpt{particle_pt_min}'
                pfn_nopid_label = f'pfn_nopid_minpt{particle_pt_min}'
                models = [pfn_charge_label, pfn_pid_label, pfn_nopid_label]

                roc_list = {}
                for model in models:
                    if model in list(self.roc_curve_dict[particle_input_type].keys()):
                        roc_list[model] = self.roc_curve_dict[particle_input_type][model]

                if self.event_type == 'dis':
                    for kappa in self.kappa:
                        charge_label = f'jet_charge_ptmin{particle_pt_min}_k{kappa}'
                        roc_list[charge_label] = self.roc_curve_dict[particle_input_type][charge_label]

                if self.class2_label == 's':
                    strange_tagger_label = f'strange_tagger_ptmin{particle_pt_min}'
                    roc_list[strange_tagger_label] = self.roc_curve_dict[particle_input_type][strange_tagger_label]

                self.plot_roc_curves(roc_list, jet_pt_min, particle_input_type, type=type, outputdir=self.output_dir_dict[particle_input_type])

            # Plot all particle_pt_min for either pid/charge/nopid
            type='varied_ptmin'
            pfn_labels = ['charge', 'pid', 'nopid']
            for pfn_label in pfn_labels:

                roc_list = {}

                for particle_pt_min in self.particle_pt_min_list:

                    model = f'pfn_{pfn_label}_minpt{particle_pt_min}'
                    if model in self.roc_curve_dict[particle_input_type].keys():
                        roc_list[model] = self.roc_curve_dict[particle_input_type][model]

                self.plot_roc_curves(roc_list, jet_pt_min, particle_input_type, type=type, outputdir=self.output_dir_dict[particle_input_type])

        #--------------------------
        # Plot combination of all possible particle_input_types in a single plot
        type = 'in_vs_out'
        pfn_labels = ['charge', 'pid', 'nopid']
        if len(list(self.roc_curve_dict.keys())) > 1:
            for pfn_label in pfn_labels:
                for particle_pt_min in self.particle_pt_min_list:

                    roc_list = {}
                    for particle_input_type in self.reference_particle_input_types:

                        model = f'pfn_{pfn_label}_minpt{particle_pt_min}'
                        if particle_input_type in self.roc_curve_dict.keys():
                            if model in self.roc_curve_dict[particle_input_type].keys():
                                roc_list[f'{model}_{particle_input_type}'] = self.roc_curve_dict[particle_input_type][model]

                    self.plot_roc_curves(roc_list, jet_pt_min, self.particle_input_type_list[0], type=type, outputdir=self.output_dir_dict[self.particle_input_type_list[0]])

    #--------------------------------------------------------------- 
    # Plot ROC curves
    #--------------------------------------------------------------- 
    def plot_roc_curves(self, roc_list, jet_pt_min, particle_input_type, type='', outputdir=''):
    
        plt.plot([0, 1], [0, 1], 'k--') # dashed diagonal
        plt.axis([0, 1, 0, 1])

        title = f'{self.class1_label} vs. {self.class2_label}     ' + rf'$p_{{\mathrm{{T,jet}}}}>{jet_pt_min}\;\mathrm{{GeV}}$'
        if type == 'fixed_ptmin':
            for label,value in roc_list.items():
                if 'pfn' in label:
                    minpt = label.rsplit('_')[2][5:]
                    if minpt == '0':
                        minpt = '0.1'
            title += f'      minpt={minpt}'
        if type == 'varied_ptmin':
            for label,value in roc_list.items():
                if 'pfn' in label:
                    if 'charge' in label:
                        title_label = '      w/charge'
                    elif 'nopid' in label:
                        title_label = '      w/o PID'
                    elif 'pid' in label:
                        title_label = '      w/PID'
            title += title_label
        if type == 'in_vs_out':
            for label,value in roc_list.items():
                input_type = label.rsplit('_')[3]
                if 'pfn' in label and input_type in ['in','leading']:
                    minpt = label.rsplit('_')[2][5:]
                    if minpt == '0':
                        minpt = '0.1'
                    if 'charge' in label:
                        title_label = '      w/charge'
                    elif 'nopid' in label:
                        title_label = '      w/o PID'
                    elif 'pid' in label:
                        title_label = '      w/PID'
                    title_label += f'      minpt={minpt}'
                    title += title_label
        if self.plot_title:
            plt.title(title, fontsize=14)

        plt.xlabel(f'False {self.class1_label} Rate', fontsize=16)
        plt.ylabel(f'True {self.class1_label} Rate', fontsize=16)
        plt.grid(True)
    
        for label,value in roc_list.items():
            index=0
            if 'pfn' in label:
                linewidth = 4
                alpha = 0.5

                minpt = label.rsplit('_')[2][5:]
                color=self.color(label, particle_pt_min=minpt, type=type)
                linestyle = self.linestyle(label)

                if type == 'fixed_ptmin':
                    if 'charge' in label:
                        label = 'Particle Flow Network (w/ charge)'
                    elif 'nopid' in label:
                        label = 'Particle Flow Network (w/o PID)'
                    elif 'pid' in label:
                        label = 'Particle Flow Network (w/ PID)'
                elif type == 'in_vs_out':
                    input_type = label.rsplit('_')[3]
                    if 'charge' in label:
                        label = 'Particle Flow Network'
                    elif 'nopid' in label:
                        label = 'Particle Flow Network'
                    elif 'pid' in label:
                        label = 'Particle Flow Network'
                    label += f', {input_type}'
                else:
                    label = 'Particle Flow Network'

            elif 'jet_charge' in label:
                linewidth = 4
                alpha = 0.5
                label = label
                linewidth = 2
                alpha = 0.6

                minpt = label.rsplit('_')[2][4:]
                kappa = label.rsplit('_')[3][1:]
                color=self.color(label, particle_pt_min=minpt, kappa=kappa, type=type)
                linestyle = self.linestyle(label)

                label = rf'Jet charge, $\kappa={kappa}$'

            elif 'strange_tagger' in label:
                linewidth = 4
                alpha = 0.5
                label = label
                linewidth = 2
                alpha = 0.6

                minpt = label.rsplit('_')[2][4:]
                color=self.color(label, particle_pt_min=minpt, type=type)
                linestyle = self.linestyle(label)

            else:
                linewidth = 2
                linestyle = 'solid'
                alpha = 0.9
                color = sns.xkcd_rgb['almost black']
  
            if type == 'varied_ptmin':
                if minpt == '0':
                    minpt = '0.1'
                label += f', min_pt = {minpt}'

            FPR = value[0]
            TPR = value[1]
            plt.plot(FPR, TPR, linewidth=linewidth, label=label,
                     linestyle=linestyle, alpha=alpha, color=color)
                    
        legend_fontsize = 10
        plt.legend(loc='lower right', fontsize=legend_fontsize)

        plt.tight_layout()
        plt.savefig(os.path.join(outputdir, f'ROC_{self.roc_plot_index[particle_input_type]}.pdf'))
        plt.close()

        self.roc_plot_index[particle_input_type] += 1

    #---------------------------------------------------------------
    # Get color for a given label
    #---------------------------------------------------------------
    def color(self, label, particle_pt_min=None, kappa=None, type=type):

        color = None

        if type == 'fixed_ptmin':

            if 'pfn_charge' in label:
                color = sns.xkcd_rgb['dark sky blue'] 
            elif 'pfn_pid' in label:
                color = sns.xkcd_rgb['faded purple']  
            elif 'pfn_nopid' in label:
                color = sns.xkcd_rgb['faded red'] 
            elif 'strange_tagger' in label:
                color = sns.xkcd_rgb['medium green']  
            elif 'jet_charge' in label:
                if kappa == '0.3':
                    color = sns.xkcd_rgb['watermelon'] 
                if kappa == '0.5':
                    color = sns.xkcd_rgb['light brown'] 
                if kappa == '0.7':
                    color = sns.xkcd_rgb['medium brown']
            else:
                color = sns.xkcd_rgb['almost black']

        elif type == 'varied_ptmin':

            if particle_pt_min == '0':
                color = sns.xkcd_rgb['faded purple'] 
            elif particle_pt_min == '0.2':
                color = sns.xkcd_rgb['faded red']    
            elif particle_pt_min == '0.4':
                color = sns.xkcd_rgb['dark sky blue']
            else:
                color = sns.xkcd_rgb['almost black']

        elif type == 'in_vs_out':

            input_type = label.rsplit('_')[3]
            if input_type in ['in', 'leading']:
                color = sns.xkcd_rgb['faded purple'] 
            elif input_type in ['out', 'leading+subleading']:
                color = sns.xkcd_rgb['light brown']    
            elif input_type in ['in+out', 'all']:
                color = sns.xkcd_rgb['medium brown'] 
            else:
                color = sns.xkcd_rgb['almost black']

        return color

    #---------------------------------------------------------------
    # Get linestyle for a given label
    #---------------------------------------------------------------
    def linestyle(self, label):
 
        linestyle = None
        if 'pfn' in label:
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