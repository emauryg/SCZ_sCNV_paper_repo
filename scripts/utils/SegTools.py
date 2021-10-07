import pybedtools as pybt
import pandas as pd
import numpy as np
import itertools as it
import pathlib2
import scipy.stats
import collections

def to_SegTool(self, sample, columns=None):
    """ Convert BedTool object to SegTool object

    Args:
        self    BedTool object

    Returns:
        st      SegTool object
    """
    # df = self.to_dataframe()
    # df.start = df.start + 1

    # if columns:
    df = self.to_dataframe(names=columns, index_col=False)
    # else:
    #     df = self.to_dataframe(index_col=False)

    st = SegTool(df, sample=sample)

    if columns is None:
        st.rename_column_(['name', 'score'], ['ratio', 'pvalue'])

    return st

# Attach to_Segtool to BedTool class
# NOTE: not bound until instantiation
pybt.BedTool.to_SegTool = to_SegTool

class SegTool(object):
    """ Class to manipulate a BICseq seg file and its derivatives.

    Attributes:
        sample  str                 name of sample
        seg     pandas data frame   segmentation data
        bed     BedTools object     seg data in bed format
    """

    def __init__(self, df, sample, renorm=None):
        self.sample = sample
        self.seg    = df
        # self.seg    = self.__parse_args(f)
        self.__check_cols()
        self.columns = self.seg.columns

        if renorm:
            self.renorm = renorm
        else:
            self.renorm = {}

    def __getitem__(self, index):
        try:
            return self.seg[index]

        except:
            if isinstance(index, slice):
                return self.seg.loc[index]

            elif isinstance(index[1], slice):
                return self.seg.iloc[index]

            else:
                return self.seg.loc[index]

    def __len__(self):
        return len(self.seg)

    @classmethod
    def read_file(cls, f, sample=None, columns=None):
        sample = f.name.split("__")[0] if not sample else sample
        df = pd.read_table(str(f), names=columns)

        return cls(df, sample)

    def __check_cols(self):
        if "log2.copyRatio" in self.seg.columns:
            self.rename_column_("log2.copyRatio", "ratio")

        if "length" not in self.seg.columns:
            self.__add_length_col()

    def __add_length_col(self):
        self.seg['length'] = self.seg.end - self.seg.start

        # Reorder columns
        cols = self.seg.columns
        cols = cols[:3].append(cols[-1:]).append(cols[3:-1])
        self.seg = self.seg[cols]

    def rename_column_(self, col_orig, col_new):
        if isinstance(col_orig, str) and isinstance(col_new, str):
            df = self.seg.rename(columns={col_orig: col_new})

        elif isinstance(col_orig, list) and isinstance(col_new, list):
            columns = dict(zip(col_orig, col_new))
            df = self.seg.rename(columns=columns)

        else:
            raise ValueError("column names are not str or list")

        self.seg = df
        self.columns = df.columns

        # return df

    def renorm_chrm_(self, chrom):
        if chrom in self.renorm:
            raise ValueError("{} has already been renormalized".format(chrom))

        if chrom not in self.seg.chrom.unique():
            raise ValueError("{} is not a chromosome in this segtool".format(chrom))

        mask = (self.seg.chrom==chrom)
        weights = self.seg.length[mask] / self.seg.length[mask].sum()
        renorm_val = np.average(self.seg.ratio[mask], weights=weights)

        self.seg.loc[mask, 'ratio'] -= renorm_val
        self.renorm[chrom] = renorm_val

        # return self

    def denorm_chrm_(self, chrom):
        if chrom not in self.seg.chrom.unique():
            raise ValueError("{} is not a chromosome in this segtool".format(chrom))

        try:
            denorm_val = self.renorm[chrom]
            mask = (self.seg.chrom==chrom)
            self.seg.loc[mask, 'ratio'] += denorm_val
            
            self.renorm.pop(chrom)

        except KeyError:
            raise ValueError("{} has not been renormalized so can't denorm it".format(chrom))

    def infer_sex_(self, chrX='chrX'):
        """ Guess the sex of the sample using chrX.
        """
        if chrX in self.renorm:
            ratio_X = self.renorm['chrX']

        else:
            df = self.seg.groupby('chrom').get_group(chrX)  
            weights = df.length / df.length.sum()
            ratio_X = np.average(df.ratio, weights=weights)

        #TODO: Should train a classifier for this, but for now quick and dirty
        if (ratio_X > -0.1) and (ratio_X < 0.1):
            sex = "female"
        elif (ratio_X > -1.1) and (ratio_X < -0.9):
            sex = "male"
        else:
            sex = "unknown"

        return sex

    def rm_chrY(self, chrX='chrX'):
        sex = self.infer_sex_(chrX)

        # print("{} is {}".format(self.sample, sex))

        if sex == "female":
            print("{} is female. Removing chromosome Y".format(self.sample))
            df = self.seg[self.seg.chrom != 'chrY']
        else:
            print("{} is {}. Returning an identical segtool".format(self.sample, sex))
            df = self.seg

        return SegTool(df, sample=self.sample, renorm=self.renorm.copy())

    def rm_sex_chrm(self):
        df = self.seg[(self.seg.chrom != 'chrY') & (self.seg.chrom != 'chrX')]

        return SegTool(df, sample=self.sample, renorm=self.renorm.copy())

    def get_breakpoints(self, pos='left'):

        keys = ['chrom', 'start', 'end']

        if pos == 'left':
            vals = [self.seg.chrom, self.seg.start, self.seg.start+1]

        if pos == 'right':
            vals = [self.seg.chrom, self.seg.end-1, self.seg.end]

        df = pd.DataFrame.from_items(zip(keys, vals))
        return SegTool(df, self.sample+'_BP_'+pos)

    def filter(self, ratio=None, pvalue=None, length_min=None, length_max=None, copy=None, renorm=[]):
        if renorm:
            for chrom in renorm:
                self.renorm_chrm_(chrom)

        df = self.seg.copy()

        if ratio:
            mask = (df['ratio'] > np.log2(1+ratio)) | (df['ratio'] < np.log2(1-ratio))
            df = df.loc[mask, :]

        if pvalue:
            df = df.loc[df['pvalue'] < pvalue, :]

        if length_min:
            df = df.loc[df['length'] > length_min, :]

        if length_max:
            df = df.loc[df['length'] < length_max, :]

        if copy == 'gain':
            df = df.loc[df['ratio'] > 0, :]
        elif copy == 'loss':
            df = df.loc[df['ratio'] < 0, :]

        return SegTool(df, sample=self.sample, renorm=self.renorm)

    def filter_batchtool(self, batchtool, A=True, f=0.5, r=False, filter_fraction=None):
        if self.bed:
            bed = self.bed
        else:
            bed = self.to_BedTool()

        if not filter_fraction:
            if r:
                F = f
            else:
                F = 1e-9

            for sample in batchtool.list_sample:
                bed = bed.subtract(batchtool[sample].bed, A=A, f=f, F=F)

        else:
            n = len(batchtool) * filter_fraction
            bed = bed.intersect(batchtool.list_bedtool, f=f, r=r, c=True)

            df = bed.to_dataframe()
            df = df[df.iloc[:, -1] < n]
            bed = pybt.BedTool.from_dataframe(df)

        return bed.to_SegTool(self.sample)

    def infer_germline(self, buffer=0.1):
        by_two = np.arange(0.5, 4, 0.5)
        upper = np.log2(by_two + 0.1)
        lower = np.log2(by_two - 0.1)

        mask = (self.seg.ratio < upper[0]) & (self.seg.ratio > lower[0])

        for u, l  in zip(upper[1:], lower[1:]):
            mask = mask | (self.seg.ratio < u) & (self.seg.ratio > l)

        df = self.seg.copy()
        df['germline'] = 0
        df.loc[mask, "germline"] = 1

        return SegTool(df, self.sample, self.renorm.copy())

    def annotate(self, bed_annot, name="annotation", nearest=False):
        ## Pure intersect and clostest. Should add distance option as well

        # Generate a unique key for each line
        bed = self.to_BedTool()
        nfields = bed.field_count()
        keys = ["\t".join(row[:nfields]) for row in bed]

        # Get intersections
        if nearest:
            bed = bed.sort()
            bed_inter = bed.closest(bed_annot, d=True)
        else:
            bed_inter = bed.intersect(bed_annot, wa=True, wb=True)

        overlaps = collections.defaultdict(set)

        for row in bed_inter:
            key = "\t".join(row[:nfields])

            if nearest:
                # overlaps[key].update(["{}".format(row[-1])])
                overlaps[key].update(["{}:{}".format(row[nfields+3], row[-1])])
            else:
                overlaps[key].update([row[nfields+3]])

        annotations = []
        for k in keys:
            if k in overlaps:
                annotations.append(",".join(sorted(overlaps[k])))
            else:
                annotations.append('')

        df = self.seg
        df[name] = annotations 

        return SegTool(df, self.sample, self.renorm.copy())

    def reciprical_overlap_(self, st_comp):
        cols_comp = st_comp.columns
        nfields = len(self.columns)

        bed = self.to_BedTool()
        bed_comp = st_comp.to_BedTool()
        nfields = bed.field_count()

        keys =  ["\t".join(row) for row in bed]
        s_ov = pd.Series(0, index=keys, dtype=float)
        bp_l = pd.Series(0, index=keys, dtype=int)
        bp_r = pd.Series(0, index=keys, dtype=int)

        bed_inter = bed.intersect(bed_comp, wao=True)
        # df_inter = bed.to_dataframe(names=self.columns+cols_comp+['overlap'])
        # df_groups = df_inter.groupby(self.columns)
        # groups = df_groups.groups()

        for row in bed_inter:
            key = "\t".join(row[:nfields])

            # overlap_score = 2*overlap / (length(feat1) + length(feat2))
            ov_score = 2 * float(row[-1]) / (int(row[2]) - int(row[1]) + int(row[nfields+2]) - int(row[nfields+1]))

            if ov_score > s_ov[key]:
                s_ov[key] = ov_score
                bp_l[key] = int(row[nfields+1])
                bp_r[key] = int(row[nfields+2])

        return s_ov, bp_l, bp_r

    def compare_segments(self, st_comp):
        """ Compare segments in this SegTool to segments in another SegTool
            by annotating each segment with the distance to the closest
            left and right breakpoint in the other SegTool and
            with the greatest reciprical overlap between a self segment
            and a segment in the other segtool

            Reciprical overlap = max_i[z_i / (x + y_i)]
                x  : segment in self
                y_i: segment in other segtool overlapping x
                z_i: bp overlap between x and y_i
                 i = 1, ..., #segments overlapping x 
        """
        bpL_comp = st_comp.get_breakpoints(pos='left').to_BedTool().sort()
        bpR_comp = st_comp.get_breakpoints(pos='right').to_BedTool().sort()

        bpL_self = self.get_breakpoints(pos='left')
        bpR_self = self.get_breakpoints(pos='right')
        
        st_bpL = bpL_self.annotate(bpL_comp, nearest=True)
        st_bpR = bpR_self.annotate(bpR_comp, nearest=True)

        overlap_score, BP_L, BP_R = self.reciprical_overlap_(st_comp)

        df = self.seg.copy()
        df['overlap'] = overlap_score.as_matrix()
        df['BP_left'] = BP_L.as_matrix()
        df['BP_right'] = BP_R.as_matrix()
        df['nearest_BP_left'] = st_bpL['annotation']
        df['nearest_BP_right'] = st_bpR['annotation']

        return SegTool(df, self.sample, self.renorm.copy())

    def mappability_score(self, bed_map):
        # Generate a unique key for each line and 
        # initiate a Series with keys as index
        bed = self.to_BedTool()
        keys = ["\t".join(row) for row in bed]
        # s_length = pd.Series(self.seg['length'], index=keys)
        s_map = pd.Series(0, index=keys)

        # Intersect map bed with self bed (assign self entry to each overlap)
        bed_inter = bed_map.intersect(bed, wb=True)
        nfields = bed_map.field_count()

        for row in bed_inter:
            key = "\t".join(row[nfields:])
            s_map[key] += len(row)

        s_map = s_map.as_matrix() / self.seg.length
        df = self.seg
        df['map_score'] = s_map

        return SegTool(df, self.sample, self.renorm.copy())

    def mapd_score(self, bin_tabix):
        """ Calculate median absolute pairwise difference for each segment
            
            Args:
                bin_tabix   BedTool     tabix indexed bed file of bins for this sample
        """
        bed = pybt.BedTool.from_dataframe(self.seg[['chrom', 'start', 'end']])

        mapd = []

        for row in bed:
            bins = bin_tabix.tabix_intervals(row)
            l2r = np.array([float(b[-1]) for b in bins])

            mapd.append(np.median(np.abs(l2r[1:] - l2r[:-1])))

        df = self.seg.copy()
        df['mapd'] = mapd
        
        return SegTool(df, self.sample, self.renorm.copy())

    def gc_content(self, fafile):
        bed = self.to_BedTool()
        nfields = bed.field_count()

        bt_nuc = bed.nucleotide_content(fi=fafile)
        # s_nuc = bt_nuc.to_dataframe().iloc[:, nfields+1]
        s_nuc = np.array([row[nfields+1] for row in bt_nuc], dtype=float)

        df = self.seg
        df['gc_content'] = s_nuc

        return SegTool(df, self.sample, self.renorm.copy())

    def to_BedTool(self, inline=False, truncate=False):
        """ Convert SegTool object to BedTool object

            kwargs:
                inline      bool        keep BedTool part of Segtool object
                truncate    bool        trim columns to chrom start end ratio pvalue
        """
        # df = self.rename_column(['ratio', 'pvalue'], ['name', 'score'])
        # df.start = df.start - 1
        df = self.seg.copy()

        if truncate:
            bed = pybt.BedTool.from_dataframe(df[['chrom', 'start', 'end', 'ratio', 'pvalue']])

        else:
            bed = pybt.BedTool.from_dataframe(df)

        if inline:
            self.bed = bed

        return bed

    def to_IGV_format(self):
        df = self.seg.copy()

        if "samples" not in df.columns:
            df['samples'] = self.sample

        df = df[["samples", "chrom", "start", "end", "ratio"]]

        return df 

    def combine(self, segtool):
        """ Combine two segtool objects
        """
        df = self.seg.append(segtool.seg, ignore_index=True).sort_values(['chrom', 'start', 'end'], axis=0)
        return SegTool(df, sample=self.sample)

    def save(self, dir_out='.', suffix='.segtool.txt'):
        f_out = (pathlib2.Path(dir_out) / self.sample).with_suffix(suffix)
        self.seg.to_csv(str(f_out), sep="\t", header=True, index=False)

    def to_grch_format(self):
        df = self.seg.copy()
        grch = df.chrom.replace('chr', '', regex=True)
        df.loc[:, 'chrom'] = grch

        return SegTool(df, self.sample, self.renorm.copy())

    def to_hg_format(self):
        df = self.seg.copy()
        hg = df.chrom.replace('^', 'chr', regex=True)
        df.loc[:, 'chrom'] = hg

        return SegTool(df, self.sample, self.renorm.copy())

class BatchSegTool(object):

    def __init__(self, segtool_list, sample_list=None, groups=None, truncate=True):
        self.list_sample = sample_list
        self.list_segtool = segtool_list
        self.truncate = truncate

        if groups:
            self.groups = groups
        else:
            self.groups = {}

        # self.__parse_input_list(input_list)
        self.__create_sample_list()
        self.__create_BedTool_list(truncate)
        self.__lists_to_dict()
        self.__create_len_list()

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.list_sample)

    def _ipython_key_completions_(self):
        return self.list_sample

    def __parse_input_list(self, input_list):
        if isinstance(input_list[0], pathlib2.Path):
            self.__read_file_list(input_list)

        elif isinstance(input_list[0], SegTool):
            self.__read_SegTool_list(input_list)

        else:
            raise TypeError("Unrecognized input of type %s" % type(input_list[0]))

    @classmethod
    def read_files(cls, input_list, sample_list=None, groups=None, columns=None, truncate=True):
        segtool_list = [SegTool.read_file(f, columns=columns) for f in input_list]

        if not sample_list:
            sample_list = [s.sample for s in segtool_list]

        if groups:
            if isinstance(groups, str):
                group_dict = {groups: sample_list.copy()}
            else:
                if len(sample_list) != len(groups):
                    raise ValueError("length of group list is different than sample list")
                group_dict = dict.fromkeys(set(groups), [])
                [group_dict[g].append(sam) for g, sam in zip(groups, sample_list)] 
        else:
            group_dict = {}

        return cls(segtool_list, sample_list, group_dict, truncate)

    def add_files(self, input_list, sample_list=None, groups=None, columns=None, truncate=True):
        bt_new = BatchSegTool.read_files(input_list, sample_list, groups, columns, truncate)
        self.list_sample.extend(bt_new.list_sample)
        self.list_segtool.extend(bt_new.list_segtool)

        if self.groups or bt_new.groups:
            self.groups.update(bt_new.groups)

        self.__lists_to_dict()

        # bt_comb = self.combine(bt_new)

    # def __read_SegTool_list(self, input_list):
    #     self.list_segtool = input_list

    def __create_sample_list(self):
        if not self.list_sample:
            self.list_sample = [s.sample for s in self.list_segtool]

    def __create_BedTool_list(self, truncate=True):
        self.list_bedtool = [s.to_BedTool(inline=True, truncate=truncate) for s in self.list_segtool]

    def __lists_to_dict(self):
        self.data = dict(zip(self.list_sample, self.list_segtool))
        # self.data = dict(it.izip(self.list_sample, self.list_segtool))

        # for sample, st, bt in it.izip(self.list_sample, self.list_segtool, self.list_bedtool):
        #     self.data[sample] = {'seg': st, 'bed': bt}

    def __create_len_list(self):
        self.list_length = [len(self[sample]) for sample in self.list_sample]

    def retrieve_group_(self, sample):
        if not self.groups:
            raise ValueError("No groups present in this batch tool")

        group = [key for key, val in self.groups.items() if sample in val]

        if len(group) > 1:
            raise ValueError("Uh oh, this {} belongs to multiple groups".format(sample))

        return group[0]

    def reduce_(self):
        df = self.list_segtool[0].seg
        df['sample'] = self.list_sample[0]

        try:
            df["group"] = self.retrieve_group_(self.list_sample[0])
        except ValueError:
            i = 1
        else:
            i = 2

        cols = df.columns

        for sample in self.list_sample[1:]:
            tmp = self[sample].seg
            tmp['sample'] = sample

            try:
                tmp["group"] = self.retrieve_group_(sample)
            except ValueError:
                print(sample)

            df = df.append(tmp, ignore_index=True)

        # cols = df.columns
        print(cols)
        cols_new = np.append(cols[-i:], cols[0:-i])
        print(cols_new)
        df = df[cols_new]

        return df

    def rm_chrY(self, chrX='chrX', auto_rm_y=False):
        segtool_list = [s.rm_chrY() for s in self.list_segtool]

        return BatchSegTool(segtool_list, self.list_sample, self.groups.copy())

    def rm_sex_chrm(self):
        segtool_list = [s.rm_sex_chrm() for s in self.list_segtool]

        return BatchSegTool(segtool_list, self.list_sample, self.groups.copy(), truncate=self.truncate)

    def samples_by_sex(self):
        sex_list = [s.infer_sex_() for s in self.list_segtool]
        female_list = [s for sex, s in zip(sex_list, self.list_segtool) if sex == 'female']
        female_samples = [s for sex, s in zip(sex_list, self.list_sample) if sex == 'female']

        male_list = [s for sex, s in zip(sex_list, self.list_segtool) if sex == 'male']
        male_samples = [s for sex, s in zip(sex_list, self.list_sample) if sex == 'male']

        bst_male = BatchSegTool(male_list, male_samples, truncate=self.truncate)
        bst_female = BatchSegTool(female_list, female_samples, truncate=self.truncate)

        return bst_male, bst_female

    def filter_by_seg_property(self, ratio=None, pvalue=None, length_min=None, length_max=None, copy=None, renorm=[], truncate=True):
        segtool_list = [s.filter(ratio, pvalue, length_min, length_max, copy, renorm) for s in self.list_segtool]
        return BatchSegTool(segtool_list, self.list_sample, self.groups.copy(), truncate=truncate)

    def filter_leave_one_out(self, sample, A=True, f=0.5, F=1e-9, r=True, filter_fraction=None):
        # print(sample)
        print("Filtering sample %s" % sample)
        bed = self[sample].bed

        if not filter_fraction:
            for s in self.list_sample:
                if s != sample:
                    # print(s)
                    bed = bed.subtract(self[s].bed, A=A, f=f, F=F)
                    # print(bed.head(n=50))

        else:
            n = len(self) * filter_fraction
            samples = [s for s in self.list_sample if s != sample]
            bed_list = [self[s].bed for s in samples]
            bed = bed.intersect(bed_list, f=f, r=r, c=True)

            df = bed.to_dataframe()
            df = df.loc[df.iloc[:, -1] < n, :]
            bed = pybt.BedTool.from_dataframe(df.iloc[:, :-1])

        return bed.to_SegTool(sample, columns=self[sample].columns)
        # return bed.to_SegTool(sample, columns=self[sample].columns)

    def filter_leave_one_out_batch(self, A=True, f=0.5, r=False, filter_fraction=None, truncate=True):
        seg_list = []

        for sample in self.list_sample:
            # print("Filtering sample %s" % sample)
            seg_list.append(self.filter_leave_one_out(sample, A=A, f=f, r=r, filter_fraction=filter_fraction))

        return BatchSegTool(seg_list, self.list_sample, self.groups.copy(), truncate=truncate)

    def filter_batchtool(self, batchtool, A=True, f=0.5, r=False, filter_fraction=None):
        seg_list = []

        for sample in self.list_sample:
            print("Filtering sample {}".format(sample))
            seg_list.append(self[sample].filter_batchtool(batchtool, A=A, f=f, r=r, filter_fraction=filter_fraction))

        return BatchSegTool(seg_list, self.list_sample)

    def merge_segments(self, distance=10000, columns=[4, 5], operation=['mean', 'min']):
        segtool_list = []
        cols = columns
        # cols = [c+1 for c in columns]

        for s in self.list_segtool:
            # print(s.bed.head())
            bed = s.bed.merge(d=distance, c=cols, o=operation)
            segtool_list.append(bed.to_SegTool(s.sample))
            # segtool_list.append(bed.to_SegTool(s.sample, s.columns[[0, 1, 2]+columns]))

        return BatchSegTool(segtool_list, self.list_sample, self.groups.copy())

    def summarize(self, f=0.9, r=True):
        cols_final = ['chrom', 'start', 'end', 'ratio', 'pvalue', 'n']
        cols_final.extend(self.list_sample)

        def _sum_rows_by_label(label, return_type=float):
        # def _sum_rows_by_label(df, label=None, return_type=float):
            # print(row)
            cols = [c for c in df.columns if label in c]
            return df[cols].astype(float).mean(axis=1).apply(return_type)

        # def _fisher_pvalue(row, cols=cols):
        def _fisher_pvalue(row):
            r = row[cols][row[cols].notnull()].as_matrix()

            if len(r) > 1:
                return scipy.stats.combine_pvalues(np.float64(r))[1]
            else:
                return r[0]

        def _count_n(row):
            return len(row[cols][row[cols].notnull()])

        def _get_chrom(row):
            for chrm in row:
                # if not np.isnan(chrm):
                try: 
                    np.isnan(chrm)
                except TypeError:
                    return chrm

        def _genome_range(row):
            if np.any(row.isnull()):
                return np.nan
            else:
                s = pd.Series([row.chrom, int(row.start), int(row.end)])
                s = s.astype(str).str.cat(sep='-')
                return s

        # Initialize empty data frame and a dummy BedTool
        cols = ["chrom", "start", "end", "ratio", "pvalue", "overlap"]
        names = [s+'_'+c for s in self.list_sample for c in cols]
        df = pd.DataFrame(columns=names)
        bt = pybt.example_bedtool('a.bed')

        # Force reciprical analysis if necessary
        if r:
            F = f
        else:
            F = 1e-9
            
        # Compare each sample to all other samples
        for sample in self.list_sample:
            print("Analyzing {}".format(sample))
            bed = self[sample].bed.subtract(bt, f=f, F=F, A=True)
            # bed = self[sample].bed.subtract(bt, f=f, r=r, A=True)

            sample_list = [s for s in self.list_sample if s is not sample]
            for s in sample_list:
                bed = bed.intersect(self[s].bed, f=f, r=r, wao=True)

            # Append bed to bt
            bt = bt.cat(bed, postmerge=False, force_trucate=True)
            # print(bt.head())

            names = [sample+"_"+val for val in ["chrom", "start", "end", "ratio", "pvalue"]]
            cols = ["chrom", "start", "end", "ratio", "pvalue", "overlap"]
            names.extend([s+"_"+c for s in sample_list for c in cols])
            df = df.append(bed.to_dataframe(names=names), ignore_index=True)

        print("Summarizing samples")
        df = df.replace(-1, np.nan)
        df = df.replace('.', np.nan)
        # return(df)

        # Chrom  start  end  ratio  std  pvalue  sample_1  sample_2  ...  sample_n
        cols = [c for c in df.columns if "chrom" in c]
        chrom  = df.apply(_get_chrom, axis=1)
        start  = _sum_rows_by_label('start', return_type=np.int64)
        end    = _sum_rows_by_label('end', return_type=np.int64)
        ratio  = _sum_rows_by_label('ratio', return_type=np.float64)

        cols = [c for c in df.columns if "pvalue" in c]
        pvalue = df.apply(_fisher_pvalue, axis=1)
        n = df.apply(_count_n, axis=1)

        vals = [chrom, start, end, ratio, pvalue, n]

        gr_list = []
        for sample in self.list_sample:
            cols = [c for c in df.columns if sample in c]
            d = df.loc[:, cols].rename(columns=dict(zip(cols, [s.split(sample+'_')[1] for s in cols])))
            d = d.loc[:, ["chrom", "start", "end"]]
            gr_list.append(d.apply(_genome_range, axis=1))

        vals.extend(gr_list)

        df_sum = pd.DataFrame.from_items(zip(cols_final, vals))
        df_sum = df_sum.sort_values(["chrom", "start", "end"], axis=0)
        return SegTool(df_sum, sample="summary")
        # return (df, pd.DataFrame.from_items(zip(cols_final, vals)))

    def save(self, dir_out='.', suffix='.segtool.out'):
        for st in self.list_segtool:
            st.save(dir_out, suffix)

    def combine(self, batchtool):
        sample_list = self.list_sample + batchtool.list_sample
        segtool_list = self.list_segtool + batchtool.list_segtool

        if self.groups or batchtool.groups:
            group_dict = self.groups.copy()
            group_dict.update(batchtool.groups)
        else:
            group_dict = None

        print(group_dict)
        return BatchSegTool(segtool_list, sample_list, group_dict) 

    def get_group(self, group, truncate=True):
        sample_list = self.groups[group]
        segtool_list = [st for st, sample in zip(self.list_segtool, self.list_sample) if sample in sample_list]

        return BatchSegTool(segtool_list, sample_list, group, truncate=truncate)

def condense_summary(df):

    def _samples_by_row(row):
        s = ', '.join([key for key, val in row.iteritems() if isinstance(val, str)])

        return s

    df_cond = df.iloc[:, 0:7]
    df_samples = df.iloc[:, 7:]

    samples = df_samples.apply(_samples_by_row, axis=1)

    df_cond['samples'] = samples

    return df_cond

