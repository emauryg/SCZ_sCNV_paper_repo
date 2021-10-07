import pysam
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
import scipy.stats

from .vcf_tools_eam import *

def plot_centromere(chrom, ax, 
                    db_cent = "/home/mas138/sandbox/dev/BICseqTools/db/hg19_centromeres-telomeres.bed"):
    df_cent = pd.read_table(db_cent, names=['chr', 'start', 'end', 'type'])
    df_cent = df_cent[df_cent['type'] == 'centromere']

    start = df_cent[df_cent.chr == chrom].start.values[0]
    end = df_cent[df_cent.chr == chrom].end.values[0]
    width = end - start
    
    ylims = ax.get_ylim()
    y0 = ylims[0]
    height = ylims[1] - ylims[0]
    
    ax.add_patch(patches.Rectangle([start, y0], width, height, color='lightgrey'))

def plot_lrr(a_pos, chrom, d, sample, ax, binsize=1, color=None, xticks=True, left=None, right=None, xlim=None, xtick_fmt=int):
    print(sample)
    lrr = np.array(d[sample][2])
    # a_pos = np.array([p for l, p in zip(d[sample][2], a_pos) if l])
    
    cp = sns.color_palette()
    if not color:
        color = cp[0]
        
    if binsize > 1:
        pos_bin, lrr_bin = bin_lrr(a_pos, chrom, d, sample, binsize)
        ax.plot(pos_bin, lrr_bin, '.', color=color)
    else:
        ax.plot(a_pos, lrr, '.', color=color)
        
    ax.hlines(0, ax.get_xlim()[0], ax.get_xlim()[1])
    
    # plot_centromere("chr{}".format(chrom), ax)
        
    ylim = ax.get_ylim()
    if left:
        ax.vlines(left, ylim[0], ylim[1])
    if right:
        ax.vlines(right, ylim[0], ylim[1])
    ax.set_ylim(ylim)
    
    if xlim:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim([0.0, max(a_pos)])
    
    if xticks:
        l = ax.get_xticks()
        ax.set_xticklabels([xtick_fmt(la / 1e6) for la in l]);
        ax.set_xlabel('chr{} (mb)'.format(chrom))
        
    ax.set_ylabel('LRR')

def plot_baf(a_pos, chrom, d, sample, ax, binsize=1, phased=True, all_snps=False, legend=True, 
             xticks=True, xlim=None, ylim=None, col_mat=None, col_pat=None,
             left=None, right=None):

    baf = np.array(d[sample][1])
    mask_gt = [gt == (1, 0) for gt in d[sample][0]]
    mask_gt2 = [gt == (0, 1) for gt in d[sample][0]]
    x_pos1 = a_pos[np.array(mask_gt) & np.array(d[sample][3])] # & (gs > 0.75)]
    y_baf1 = baf[np.array(mask_gt) & np.array(d[sample][3])] # & (gs > 0.75)]
    x_pos2 = a_pos[np.array(mask_gt2) & np.array(d[sample][3])] # & (gs > 0.75)]
    y_baf2 = baf[np.array(mask_gt2) & np.array(d[sample][3])] # & (gs > 0.75)]
    
    cp = sns.color_palette()
    if not col_mat:
        col_mat = cp[2]
    if not col_pat:
        col_pat = cp[4]

    if all_snps:
        baf = np.array(d[sample][1])
        ax.plot(a_pos, baf, '.', color=col_pat, label='BAF all SNPs')
        
    elif binsize > 1:
        pos_bin, baf_bin, pos, baf_ref = bin_snps(a_pos, chrom, d, sample, binsize)
        ax.plot(pos_bin, baf_bin, '.', color=col_pat, label="{}:{}-{}".format(sample, left, right))

    elif phased:
        ax.plot(x_pos1, 1-y_baf1, '.', color=col_pat, label='paternal ALT')
        ax.plot(x_pos2, y_baf2, '.', color=col_pat, label='paternal REF')


    else:
        ax.plot(x_pos2, y_baf2, '.', color=col_mat, label='maternal')
        ax.plot(x_pos1, y_baf1, '.', color=col_pat, label='paternal')
    
    if legend:
        ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, borderaxespad=0.)
       
    if ylim:
        ax.set_ylim(ylim)
    # ax.set_ylim([0, 1])
    
    if xlim:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim([0.0, np.max(a_pos)])
        # ax.set_xlim([0.0, max(max(x_pos1), max(x_pos2))])
        
    ax.hlines(0.66, ax.get_xlim()[0], ax.get_xlim()[1], linestyles='--')
    ax.hlines(0.33, ax.get_xlim()[0], ax.get_xlim()[1], linestyles='--')
    ax.hlines(0.5, ax.get_xlim()[0], ax.get_xlim()[1])
    
    # plot_centromere("chr{}".format(chrom), ax)
        
    ylim = ax.get_ylim()
    if left:
        ax.vlines(left, ylim[0], ylim[1])
    if right:
        ax.vlines(right, ylim[0], ylim[1])
    ax.set_ylim(ylim)
    
    if xticks:
        l = ax.get_xticks()
        ax.set_xticklabels([int(la / 1e6) for la in l]);
        ax.set_xlabel('chr{} (mb)'.format(chrom))
        
    ax.set_ylabel('PAF')

def plot_event(SAMPLE, CHROM, BEG, END, binsize=1, stretch=2):

    LEN = END - BEG
    start = BEG - stretch * LEN
    if start < 0:
        start = 0
    end = END + stretch * LEN

    vcf = sample_to_VCF(SAMPLE)
    pos_list, d = fetch_snps(vcf, CHROM, [SAMPLE], start=start, end=end)
    
    a_pos = np.array(pos_list)

    f, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    plot_baf(a_pos, CHROM, d, SAMPLE, ax[0], binsize=binsize, left=BEG, right=END, xlim = [start, end])
    plot_lrr(a_pos, CHROM, d, SAMPLE, ax[1], binsize=binsize, left=BEG, right=END, xlim = [start, end])
    ax[0].set_xlabel('')
    ax[0].legend('')
    
    ymin, ymax = ax[1].get_ylim()
    if ymin < -0.5:
        ymin = -0.5
    if ymax > 0.5:
        ymax = 0.5
    ax[1].set_ylim([ymin, ymax])
    plt.tight_layout()

def plot_chromosome(SAMPLE, CHROM, binsize=16, BEG=None, END=None, vcf=None, all_snps=False, figsize=(12, 7)):

    if not vcf:
        vcf = sample_to_VCF(SAMPLE)

    pos_list, d = fetch_snps(vcf, CHROM, [SAMPLE])
    
    a_pos = np.array(pos_list)

    if not BEG:
        BEG = a_pos[0]
    if not END:
        END = a_pos[-1]

    f, ax = plt.subplots(2, 1, figsize=figsize, sharex=True)
    plot_baf(a_pos, CHROM, d, SAMPLE, ax[0], binsize=binsize, left=BEG, right=END, all_snps=all_snps)
    plot_lrr(a_pos, CHROM, d, SAMPLE, ax[1], binsize=binsize, left=BEG, right=END)
    ax[0].set_xlabel('')
    ax[0].legend('')
    
    ymin, ymax = ax[1].get_ylim()
    if ymin < -0.5:
        ymin = -0.5
    if ymax > 0.5:
        ymax = 0.5
    ax[1].set_ylim([ymin, ymax])
    plt.tight_layout()

