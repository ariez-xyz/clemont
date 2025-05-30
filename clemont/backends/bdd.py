import sys
import json
from collections import defaultdict
from .discretization import Discretization
from .base import BaseBackend

try:
    from dd.cudd import BDD as cuddMgr
    _HAS_CUDD = True
except ImportError:
    _HAS_CUDD = False
    cuddMgr = None

BAD_CHARS = {
    '=': 'eq',
    '+': 'plus',
    '<': 'lt',
    '>': 'gt',
    '/': 'slash',
    '(': '',
    '-': '',
    ')': '',
    '&': '',
    '|': '',
    '^': '',
    ' ': '',
    '_': '',
}

class BDD(BaseBackend):
    def __init__(self, data_sample, n_bins, decision_col, onehot_cols=[], categorical_cols=[], collect_cex=False):

        if not _HAS_CUDD:
            raise ImportError(
                "\n" + "="*60 + "\n"
                "BDD backend requires 'dd' package with CUDD bindings.\n"
                "This cannot be installed automatically via pip.\n\n"
                "Please see the installation instructions in the README.\n"
                "\n" + "="*60 + "\n"
            )

        self.collect_cex = collect_cex

        if any(any(char in col for char in BAD_CHARS.keys()) for col in data_sample.columns):
            raise ValueError(f"BDD error: column names cannot contain characters {BAD_CHARS.keys()}, please replace them (e.g. using BDD.BAD_CHARS)")

        self.discretization = Discretization(data_sample, n_bins, decision_col, onehot_cols, categorical_cols)
        self.id_map = defaultdict(list)

        self.bdd = cuddMgr()
        self.bdd.declare(*self.discretization.bdd_vars)

        self.history_bdd = self.bdd.false
        
        self.fairness_bdd = self.Linf_bdd(
                self.bdd, 
                self.discretization.cols, 
                self.discretization.decision, 
                self.discretization.vars_map,
                distances={cat: 0 for cat in self.discretization.categorical_cols} # Categories must be exact match, others count as similar if they are in neighboring bins
            )

        self.bdd.configure(reordering=False) # must be done after fairness bdd or it'll be very slow

        self._meta = {
            "metric": "Linf",
            "epsilon": 1/n_bins,
            "decision_col": decision_col,
            "is_exact": True,
            "is_sound": False,
            "is_complete": True,
            "n_vars": len(self.discretization.bdd_vars),
        }

    def hash_dict(self, d): # There may be better options but this is not as bad as it seems
        return json.dumps({k:v for k,v in d.items() if k.startswith('x_')}, sort_keys=True)

    def index(self, df):
        for index, row in df.iterrows():
            binned_row = self.discretization.bin_row(row)
            x_val, _ = self.make_valuations(binned_row)
            self.history_bdd = self.history_bdd | self.bdd.cube(x_val) # add current sample to history
            if index % 10000 == 0:
                print(index)

    def observe(self, row, row_id=None):
        if self.collect_cex and row_id == None:
            self.collect_cex = False
            print("warn: need a row_id to collect multiple counterexamples. disabling counterexample collection", file=sys.stderr)

        binned_row = self.discretization.bin_row(row)
        x_val, y_val = self.make_valuations(binned_row)
        
        if self.collect_cex:
            self.id_map[self.hash_dict(x_val)].append(row_id)
        
        self.history_bdd = self.history_bdd | self.bdd.cube(x_val) # add current sample to history

        E = self.fairness_bdd & self.bdd.cube(y_val)
        E = E & self.history_bdd
        is_fair = (E == self.bdd.false)

        # Counterexample is provided by Cudd in the form of a valuation.
        # First collect all violating valuations
        cex_valuations = []
        if not is_fair:
            #if self.collect_cex == "one": # Faster. TODO: What to do about no-cex monitoring support? Since we have so many false positives.
            #    cex_valuations = [self.bdd.pick(E)]
            if self.collect_cex:
                cex_valuations += [cex for cex in self.bdd.pick_iter(E, care_vars=self.discretization.bdd_vars)]

        # Then map to indices
        cex_indices = self.to_indices(cex_valuations)

        return cex_indices

    # to_indices maps list of valuations to the indices of the points situated within the discrete cells represented by the valuations
    def to_indices(self, valuations):
        ids = set()
        for v in valuations:
            ids |= set(self.id_map[self.hash_dict(v)])
        return ids
                
    def make_valuations(self, row):
        ret = []
        for p in 'xy':
            d = {}
            t = self.discretization.vars_map[p]
            for col in t.keys():
                for i, var in enumerate(t[col]):
                    if row[col] >> i & 1:
                        d[var] = True
                    else:
                        d[var] = False
            ret += [d]
        return ret

    # Create BDD over 2n variables. BDD should be True if the two inputs represent a biased decision
    # That can mean anything you want it to. A practical limitation is being able to build a BDD from the map (enumerating all input pairs is quickly infeasible)
    def Linf_bdd(self, bdd, cols, decision, vars_map, distances={}):
        D = bdd.true
        warned = False
        
        def same(col, j, nbits):
            return f'(( x_{col}_{nbits-j-1} &  y_{col}_{nbits-j-1}) | (!x_{col}_{nbits-j-1} & !y_{col}_{nbits-j-1}))'
        def diff(col, j, nbits):
            return f'(( x_{col}_{nbits-j-1} & !y_{col}_{nbits-j-1}) | (!x_{col}_{nbits-j-1} &  y_{col}_{nbits-j-1}))'
        def one(var, col, j, nbits):
            return f' {var}_{col}_{nbits-j-1}'
        def zero(var, col, j, nbits):
            return f'!{var}_{col}_{nbits-j-1}'
        
        
        for col in cols:
            col_flas = []
            col_nbits = len(vars_map['x'][col])
        
            # same-case
            col_flas.append(f"({' & '.join([same(col, j, col_nbits) for j in range(col_nbits)])})")

            if col == decision: # decision bit must always be different
                D = D & ~bdd.add_expr(col_flas[0])
                continue
            
            # default: neighboring bins count as similar
            if col not in distances.keys() or distances[col] == 1:
                for order in [('x', 'y'), ('y', 'x')]:
                    for k in range(col_nbits):
                        terms = []
                
                        # first k-1 bits match
                        for j in range(k):
                            terms.append(same(col, j, col_nbits))
                
                        # first has suffix 100000... 
                        terms.append(one(order[0], col, k, col_nbits))
                        for j in range(k+1, col_nbits):
                            terms.append(zero(order[0], col, j, col_nbits))
                
                        # second has suffix 011111...
                        terms.append(zero(order[1], col, k, col_nbits))
                        for j in range(k+1, col_nbits):
                            terms.append(one(order[1], col, j, col_nbits))
                        
                        #print("\n","\n      &\t".join(terms), sep='\t')
                        col_flas.append(f'({" & ".join(terms)})')

            if col in distances.keys() and distances[col] not in {0,1}:
                if not warned:
                    warned = True
                    print(f'warning: currently similarity BDD only supports distances 0,1! defaulted {col} to 0.', file=sys.stderr)
                
            # at this point col_flas contains a list of ways in that  | x_col - y_col | <= distances[col]
            # OR these into a single fla
            col_matches = f"({'   |   '.join(col_flas)})"

            # the similarity BDD wants every col to be close
            # An error here is likely because 
            # - the column is assigned 0 variables.
            # - column naming issues, e.g. ab_c and a_bc are the same after mapping
            # Uncomment the below and check vars_map
            # print(vars_map, col, col_flas, col_nbits)
            D = D & bdd.add_expr(col_matches)

        return D

