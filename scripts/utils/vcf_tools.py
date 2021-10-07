import pysam
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
import scipy.stats

## HARD-CODED data paths
# load proband data
df_p1 = pd.read_table("/n/no_backup2/dbmi/park/max/SSC/Microarray/UCSF/mocha/1Mv1/1Mv1.mocha.probands.tsv")
df_p2 = pd.read_table("/n/no_backup2/dbmi/park/max/SSC/Microarray/UCSF/mocha/1Mv3/1Mv3.mocha.probands.tsv")
df_p3 = pd.read_table("/n/no_backup2/dbmi/park/max/SSC/Microarray/UCSF/mocha/Omni2.5/Omni2.5.mocha.probands.tsv")

# Siblings data
df_s1 = pd.read_table("/n/no_backup2/dbmi/park/max/SSC/Microarray/UCSF/mocha/1Mv1/1Mv1.mocha.siblings.tsv")
df_s2 = pd.read_table("/n/no_backup2/dbmi/park/max/SSC/Microarray/UCSF/mocha/1Mv3/1Mv3.mocha.siblings.tsv")
df_s3 = pd.read_table("/n/no_backup2/dbmi/park/max/SSC/Microarray/UCSF/mocha/Omni2.5/Omni2.5.mocha.siblings.tsv")

# Parent data
df_md1 = pd.read_table("/n/no_backup2/dbmi/park/max/SSC/Microarray/UCSF/mocha/1Mv1/1Mv1.mocha.parents.tsv")
df_md2 = pd.read_table("/n/no_backup2/dbmi/park/max/SSC/Microarray/UCSF/mocha/1Mv3/1Mv3.mocha.parents.tsv")
df_md3 = pd.read_table("/n/no_backup2/dbmi/park/max/SSC/Microarray/UCSF/mocha/Omni2.5/Omni2.5.mocha.parents.tsv")

df_1Mv1 = pd.concat([df_p1, df_s1, df_md1])
df_1Mv3 = pd.concat([df_p2, df_s2, df_md2])
df_omni = pd.concat([df_p3, df_s3, df_md3])
df_reg  = pd.read_table("/n/no_backup2/dbmi/park/max/SPARK/Regeneron/SPARK.30K.Release.genotype.samples", names=['SAMPLE'])

# load vcfs
vcf_1Mv1 = pysam.VariantFile("/n/no_backup2/dbmi/park/max/SSC/Microarray/UCSF/VCFs/1Mv1/1Mv1.phased.bcf")
vcf_1Mv3 = pysam.VariantFile("/n/no_backup2/dbmi/park/max/SSC/Microarray/UCSF/VCFs/1Mv3/1Mv3.phased.bcf")
vcf_Omni = pysam.VariantFile("/n/no_backup2/dbmi/park/max/SSC/Microarray/UCSF/VCFs/Omni2.5//Omni2.5.phased.bcf")
# vcf_reg  = pysam.VariantFile("/n/no_backup2/dbmi/park/max/SPARK/Regeneron/tests/plink2vcf_test/first_1000/mocha_LRRnorm_2019-03-24/first_1000.annot.fixref.LRRnorm.mocha.bcf")
vcf_reg  = pysam.VariantFile("/n/no_backup2/dbmi/park/max/SPARK/Regeneron/plink2vcf/regeneron.annot.dedup.LRRnorm.phased.bcf")
# vcf_reg  = pysam.VariantFile("/n/no_backup2/dbmi/park/max/SPARK/Regeneron/plink2vcf/regeneron.annot.dedup.phased.bcf")

def fetch_snps(vcf, chrom, samples, start=None, end=None):
    pos_list = []
    rs_list = []
    d = {item: [[], [], [], []] for item in samples}

    for snp in vcf.fetch(chrom, start=start, end=end):
        pos_list.append(snp.pos)
        rs_list.append(snp.id)
        for sample in samples:
            rec = snp.samples.get(sample)
            d[sample][0].append(rec.get('GT'))
            d[sample][1].append(rec.get('BAF'))
            d[sample][2].append(rec.get('LRR'))
            d[sample][3].append(rec.phased)
            
    return pos_list, d

def bin_snps(a_pos, chrom, d, sample, bin_size):
    """ Bin neighboring SNPs together
        Unifies the SNPs to a single haplotype
    """
    mask_pat = np.array([gt == (0, 1) for gt in d[sample][0]])
    mask_mat = np.array([gt == (1, 0) for gt in d[sample][0]])
    mask_het = mask_pat | mask_mat
    mask_ph = np.array(d[sample][3])
    
    pos = a_pos[mask_het & mask_ph] 
    baf_alt = np.array(d[sample][1])
    baf_pat = baf_alt.copy()
    baf_pat[mask_mat] = 1 - baf_alt[mask_mat]
    baf_ph = baf_pat[mask_het & mask_ph]
    
    idx = np.concatenate([np.arange(0, len(baf_ph)-1, bin_size), [len(baf_ph)-1]])
    baf_bin = np.array([np.median(baf_ph[i:j]) for i, j in zip(idx[:-1], idx[1:])])
    a_pos_bin = np.array([pos[i] for i in idx[:-1]])
    
    return a_pos_bin, baf_bin, pos, baf_ph

def bin_lrr(a_pos, chrom, d, sample, bin_size):
    """ Bin LRR from neighboring SNPs together
        Unifies the SNPs to a single haplotype
    """
    # lrr = np.array(d[sample][2])
    lrr = np.array([l for l in d[sample][2] if l])
    a_pos = np.array([p for l, p in zip(d[sample][2], a_pos) if l])

    idx = np.concatenate([np.arange(0, len(lrr)-1, bin_size), [len(lrr)-1]])
    lrr_bin = np.array([np.median(lrr[i:j]) for i, j in zip(idx[:-1], idx[1:])])
    a_pos_bin = np.array([a_pos[i] for i in idx[:-1]])
    
    return a_pos_bin, lrr_bin

def sample_to_VCF(sample):
    if (df_1Mv1.SAMPLE == sample).any():
        vcf = vcf_1Mv1
    elif (df_1Mv3.SAMPLE == sample).any():
        vcf = vcf_1Mv3
    elif (df_omni.SAMPLE == sample).any():
        vcf = vcf_Omni
    elif (df_reg.SAMPLE == sample).any():
        vcf = vcf_reg
    else:
        raise ValueError("{} not found in any experiment set.".format(sample))

    return vcf
