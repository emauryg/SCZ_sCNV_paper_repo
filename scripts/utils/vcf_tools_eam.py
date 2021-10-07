import pysam
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
import scipy.stats

## HARD-CODED data paths
# load study data
all_study_data =  pd.read_table('/home/pgccnv/somaticcnv/IllumOmniExpress/IllumOmniExpress_sample_info_20190603.txt')
all_study_data = all_study_data.replace(".txt","",regex=True)
df_ms_scz_uktr_eur_qc=  all_study_data[all_study_data.dataset == "ms.scz_uktr_eur-qc"]
df_scz_clm2_eur_qc_1 = all_study_data[all_study_data.dataset == "scz_clm2_eur-qc_1"]
df_scz_cimsb_eur_qc = all_study_data[all_study_data.dataset == "scz_cimsb_eur-qc"]
df_scz_clm2_eur_qc = all_study_data[all_study_data.dataset == "scz_clm2_eur-qc"]
df_scz_clo3_eur_qc = all_study_data[all_study_data.dataset == "scz_clo3_eur-qc"]
df_scz_cou3_eur_qc_1 = all_study_data[all_study_data.dataset == "scz_cou3_eur-qc_1"]
df_scz_cou3_eur_qc_3 = all_study_data[all_study_data.dataset == "scz_cou3_eur-qc_3"]
df_scz_egcu_eur_qc = all_study_data[all_study_data.dataset == "scz_egcu_eur-qc"]
df_scz_umebs_eur_qc = all_study_data[all_study_data.dataset == "scz_umebs_eur-qc"]
df_sweden6 = all_study_data[all_study_data.dataset == "sweden6"]
df_scz_ersw_eur_qc = all_study_data[all_study_data.dataset == "scz_ersw_eur-qc"]
df_scz_clo3_eur_qc_1 = all_study_data[all_study_data.dataset == "scz_clo3_eur-qc_1"]
df_sweden5 = all_study_data[all_study_data.dataset == "sweden5"]


# load vcfs
vcf_ms_scz_uktr_eur_qc = pysam.VariantFile("/home/pgccnv/somaticcnv/mocha_output/ms.scz_uktr_eur-qc.mocha.bcf")
vcf_scz_clm2_eur_qc_1 = pysam.VariantFile("/home/pgccnv/somaticcnv/mocha_output/scz_clm2_eur-qc_1.mocha.bcf")
vcf_scz_cimsb_eur_qc = pysam.VariantFile("/home/pgccnv/somaticcnv/mocha_output/scz_cimsb_eur-qc.mocha.bcf")
vcf_scz_clm2_eur_qc = pysam.VariantFile("/home/pgccnv/somaticcnv/mocha_output/scz_clm2_eur-qc.mocha.bcf")
vcf_scz_clo3_eur_qc = pysam.VariantFile("/home/pgccnv/somaticcnv/mocha_output/scz_clo3_eur-qc.mocha.bcf")
vcf_scz_cou3_eur_qc_1 = pysam.VariantFile("/home/pgccnv/somaticcnv/mocha_output/scz_cou3_eur-qc_1.mocha.bcf")
vcf_scz_cou3_eur_qc_3 = pysam.VariantFile("/home/pgccnv/somaticcnv/mocha_output/scz_cou3_eur-qc_3.mocha.bcf")
vcf_E26 = pysam.VariantFile("/home/pgccnv/somaticcnv/mocha_output/E26.mocha.bcf")
vcf_OmniTE = pysam.VariantFile("/home/pgccnv/somaticcnv/mocha_output/OmniTE.mocha.bcf")
vcf_scz_ersw_eur_qc = pysam.VariantFile("/home/pgccnv/somaticcnv/mocha_output/scz_ersw_eur-qc.mocha.bcf")
vcf_sweden6 = pysam.VariantFile("/home/pgccnv/somaticcnv/mocha_output/sweden6.mocha.bcf")
vcf_scz_umebs_eur_qc = pysam.VariantFile("/home/pgccnv/somaticcnv/mocha_output/scz_umebs_eur-qc.mocha.bcf")
vcf_scz_clo3_eur_qc_1 = pysam.VariantFile("/home/pgccnv/somaticcnv/mocha_output/scz_clo3_eur-qc_1.mocha.bcf")
vcf_sweden5 = pysam.VariantFile("/home/pgccnv/somaticcnv/mocha_output/sweden5.mocha.bcf")

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
	if (df_ms_scz_uktr_eur_qc.alt_identifier == sample).any():
		vcf = vcf_ms_scz_uktr_eur_qc
	elif (df_scz_clm2_eur_qc_1.alt_identifier == sample).any():
		vcf = vcf_scz_clm2_eur_qc_1
	elif (df_scz_cimsb_eur_qc.alt_identifier == sample).any():
		vcf = vcf_scz_cimsb_eur_qc
	elif (df_scz_clm2_eur_qc.alt_identifier == sample).any():
		vcf = vcf_scz_clm2_eur_qc
	elif (df_scz_clo3_eur_qc.alt_identifier == sample).any():
		vcf = vcf_scz_clo3_eur_qc
	elif (df_scz_clo3_eur_qc_1.alt_identifier == sample).any():
		vcf = vcf_scz_clo3_eur_qc_1
	elif (df_scz_cou3_eur_qc_1.alt_identifier == sample).any():
		vcf = vcf_scz_cou3_eur_qc_1
	elif (df_scz_cou3_eur_qc_3.alt_identifier == sample).any():
		vcf = vcf_scz_cou3_eur_qc_3
	elif (np.array(vcf_scz_ersw_eur_qc.header.samples).astype(str) == sample).any():
		vcf = vcf_scz_ersw_eur_qc
	elif (df_sweden6.alt_identifier == sample).any():
		vcf = vcf_sweden6
	elif (df_sweden5.alt_identifier == sample).any():
		vcf = vcf_sweden5
	elif (df_scz_umebs_eur_qc.alt_identifier == sample).any():
		vcf = vcf_scz_umebs_eur_qc
	elif (df_scz_egcu_eur_qc.identifier == sample).any():
		if (np.array(vcf_E26.header.samples) == sample).any():
			vcf=vcf_E26
		elif (np.array(vcf_OmniTE.header.samples) == sample).any():
			vcf = vcf_OmniTE
		else:
			raise ValueError("{} not found in scz_egcu_eur-qc.".format(sample))
	else:
		raise ValueError("{} not found in any experiment set.".format(sample))
	return vcf


def em_assign_ML2(df, b_dup, b_del, s_dup, s_del, s_loh):
    
    def ml(row):
        if row.LDEV > 0:
            ll_dup = scipy.stats.norm.pdf(row.LDEV, row.bdev*b_dup, s_dup)
            ll_loh = scipy.stats.norm.pdf(row.LDEV, 0, s_dup)
            ll_del = 0
        else:
            ll_del = scipy.stats.norm.pdf(row.LDEV, row.bdev*b_del, s_del)
            ll_loh = scipy.stats.norm.pdf(row.LDEV, 0, s_del)
            ll_dup = 0
        
        return (ll_dup, ll_del, ll_loh)
    
    def assign(row):
        if row.LDEV > 0:
            if row.p_DUP > row.p_LOH:
                return 'DUP'
            else:
                return 'CNN-LOH'
        else:
            if row.p_DEL > row.p_LOH:
                return 'DEL'
            else:
                return 'CNN-LOH'
    
    ll = df.apply(ml, axis=1)
    df['p_DUP'] = [l[0] for l in ll]
    df['p_LOH'] = [l[2] for l in ll]
    df['p_DEL'] = [l[1] for l in ll]
    CLASS = df.apply(assign, axis=1)
    
    return CLASS

def em_assign_ML3(df, b_dup, b_del, s_dup, s_del, s_loh):
    ## P_ARM, Q_ARM
    ## Prior probabilities with Telomere
    pr_dup_T = 0.1906
    pr_del_T = 0.0949
    pr_loh_T = 0.7145
    ## Prior probabilities without Telomere
    pr_dup_noT = 0.0786
    pr_del_noT = 0.8846
    pr_loh_noT = 0.0368
    def ml(row):
        if (row.p_arm == 'T' or row.q_arm == 'T') :
            if row.LDEV > 0:
                ll_dup = pr_dup_T*scipy.stats.norm.pdf(row.LDEV, row.bdev*b_dup, s_dup)
                ll_loh = pr_loh_T*scipy.stats.norm.pdf(row.LDEV, 0, s_dup)
                ll_del = 0
            else:
                ll_del = pr_del_T*scipy.stats.norm.pdf(row.LDEV, row.bdev*b_del, s_del)
                ll_loh = pr_loh_T*scipy.stats.norm.pdf(row.LDEV, 0, s_del)
                ll_dup = 0
        else:
            if row.LDEV > 0:
                ll_dup = pr_dup_noT*scipy.stats.norm.pdf(row.LDEV, row.bdev*b_dup, s_dup)
                ll_loh = pr_loh_noT*scipy.stats.norm.pdf(row.LDEV, 0, s_dup)
                ll_del = 0
            else:
                ll_del = pr_del_noT*scipy.stats.norm.pdf(row.LDEV, row.bdev*b_del, s_del)
                ll_loh = pr_loh_noT*scipy.stats.norm.pdf(row.LDEV, 0, s_del)
                ll_dup = 0
        return (ll_dup, ll_del, ll_loh)
    def assign(row):
        if row.LDEV > 0:
            if row.p_DUP > row.p_LOH:
                return 'DUP'
            else:
                return 'CNN-LOH'
        else:
            if row.p_DEL > row.p_LOH:
                return 'DEL'
            else:
                return 'CNN-LOH'
    ll = df.apply(ml, axis=1)
    df['p_DUP'] = [l[0] for l in ll]
    df['p_LOH'] = [l[2] for l in ll]
    df['p_DEL'] = [l[1] for l in ll]
    CLASS = df.apply(assign, axis=1)
    return CLASS


def em_slope_ML(df):
    df_dup = df[df.CLASS == 'DUP']
    df_del = df[df.CLASS == 'DEL']
    
    b_dup = (df_dup.bdev * df_dup.LDEV).sum() / (df_dup.bdev**2).sum()
    b_del = (df_del.bdev * df_del.LDEV).sum() / (df_del.bdev**2).sum()
    
    s_dup = (df[df.CLASS == 'DUP'].LDEV - df[df.CLASS == 'DUP'].bdev * b_dup).std(ddof=1)
    s_del = (df[df.CLASS == 'DEL'].LDEV - df[df.CLASS == 'DEL'].bdev * b_del).std(ddof=1)
    s_loh = (df[df.CLASS == 'CNN-LOH'].LDEV).std(ddof=1)
    
    return b_dup, b_del, s_dup, s_del, s_loh
def em_run_ML(df):
    df['CLASS'] = ''
    # 1. Randomly pick params (kind of)
    b_dup_em = np.random.rand(1)[0] * 3
    b_del_em = np.random.rand(1)[0] * -3
    b_loh_em = 0.
    s_dup_em = 0.01
    s_del_em = 0.01
    s_loh_em = 0.01
    # 2. Assign points to line 
    # CLASS = em_assign_ML(df, b_dup_em, b_del_em, s_dup_em, s_del_em, s_loh_em)
    # CLASS = em_assign_ML2(df, b_dup_em, b_del_em, s_dup_em, s_del_em, s_loh_em)
    CLASS = em_assign_ML3(df, b_dup_em, b_del_em, s_dup_em, s_del_em, s_loh_em)	
    # Iterate until convergence
    i = 0
    while np.any(CLASS != df.CLASS):
        print(i)
        i += 1
        df.CLASS = CLASS
        b_dup_em, b_del_em, s_dup_em, s_del_em, s_loh_em = em_slope_ML(df)
        # CLASS = em_assign_ML(df, b_dup_em, b_del_em, s_dup_em, s_del_em, s_loh_em)
        CLASS = em_assign_ML3(df, b_dup_em, b_del_em, s_dup_em, s_del_em, s_loh_em)   
    return b_dup_em, b_del_em, s_dup_em, s_del_em, s_loh_em
