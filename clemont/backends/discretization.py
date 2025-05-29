import numpy as np
import sys

class Discretization:
    # order of calls matters
    def __init__(self, df, n_bins, decision_col, onehot_cols=[], categorical_cols=[]):
        if onehot_cols != []:
            for onehot in onehot_cols:
                if type(onehot) is not set:
                    raise ValueError("onehot_cols must be a list of sets each specifying a group of columns that together represent a single one-hot encoded attribute")
                categorical_cols += list(onehot)
            # We can save bits and thus variables by undoing the one-hot encoding
            # leave this optimization for later, needs to be implemented in transform_row
            print("warning: treating onehot cols as categorical cols", file=sys.stderr)

        self.raw_cols = list(map(str, df.columns))
        self.onehot_cols = onehot_cols
        self.categorical_cols = categorical_cols
        self.decision = decision_col
        self.n_bins = n_bins
        self.cols = self.raw_cols

        self.col_counts = {col: df[col].nunique() for col in self.raw_cols}
        self.cols_to_bin = [col for col in self.raw_cols if col not in categorical_cols]

        self.bins = self.make_bins(df)
        self.bdd_vars = self.make_bdd_vars()
        self.vars_map = self.make_vars_map()


    # Transform and bin a row
    def bin_row(self, row):
        # TODO: 
        # 1. Turn categorical cols into contiguous ints.
        # 2. Bin numerical cols. (this is done)
        # 3. Undo onehot encoding, this should also be done here.
        row = row.copy()
        for col, bin_edges in self.bins.items():
            if row[col] == bin_edges[-1]: # np.digitize doesn't count last bin as inclusive
                row[col] = len(bin_edges) - 2
            else:
                row[col] = np.digitize(row[col], bin_edges) - 1
        return row.astype(int)

    # NOTE: This breaks if we get values outside the range of the bins.
    # Either assume the range of the features must be known beforehand. Then it probably makes more sense to bin this by range instead of min/max of the df.
    # Or don't and clip values, or add two outermost bins that stretch to +-inf. The latter is handled cleanly by np.digitize! e.g. np.digitize([1,2,3], [1.5, 2.5]) => 0, 1, 2.
    def make_bins(self, df):
        bins = {}
        for col in self.cols_to_bin:
            # TODO: Is this sensible at all? Categorical (low uniq-count) cols are excluded prior to this. Maybe just bin them no questions asked?
            # TODO: Use n_bins-2 so that we can handle new values outside the min/max of the df? Or assume range is known beforehand?
            cur_nbins = min(self.n_bins, self.col_counts[col] - 1)

            if col == self.decision:
                cur_nbins =  self.col_counts[col]
                print(f"BDD: Assuming {cur_nbins} decision classes", file=sys.stderr)

            bin_edges = np.histogram_bin_edges(df[col], bins=cur_nbins)
            bins[col] = bin_edges

        return bins

    def make_bdd_vars(self):
        bdd_vars = []
        for j in ['x', 'y']:
            for col in self.cols:
                col_nunique = self.col_counts[col] # TODO: This only makes sense if the values are contiguous!!!! It should be max of col. But should be fine for 
                if col in self.bins.keys():
                    col_nunique = len(self.bins[col]) - 1
                col_nbits = np.ceil(np.log2(col_nunique)).astype(int)
                bdd_vars += [f'{j}_{col}_{i}' for i in range(col_nbits)]
                
        return bdd_vars

    def make_vars_map(self):
        tx, ty = {}, {}
        for col in self.cols:
            tx[col] = []
            ty[col] = []
            for entry in self.bdd_vars:
                if entry.startswith("x_" + col + "_"): # these checks aren't totally bulletproof, eg one column called day, another day_time
                    tx[col] += [entry]
                elif entry.startswith("y_" + col + "_"):
                    ty[col] += [entry]
        return {'x': tx, 'y': ty }

