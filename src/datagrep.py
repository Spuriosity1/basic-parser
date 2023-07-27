# Class for storing and parsing moderately large datasets
import os.path as path
import os
import numpy as np
import toml, json

BASIC_TYPES = [str, int, float, bool]

def comprehend_parameters(spec) -> dict[str, int | float | str ]:
    """
    Accpts either a pathlike route to a toml paramfile
    formatted 'python style', where decimals are always read as floats
    nreps = 2
    special syntax:
    gprime = @float
    for generic (but unspecified) floating point
    @ literals at the start of a string param must be added with `@@`
    """
    if type(spec) is dict:
        return spec
    
    root, ext = path.splitext(spec)

    val = None
    if ext == '.toml':
        val = toml.load(spec)
    elif ext == '.json':
        val = json.load(spec)
    else:
        raise NotImplementedError("File extention "+ext+" is not recognised.")

    for p in val:
        # assert shallow
        if type(val[p]) not in BASIC_TYPES:
            raise ValueError(f"Parameter {root}[{p}] = {val[p]} has illegal type")
        if type(val[p]) is str:
            if val[p].startswith('@'):
                after = val[p][1:]
                if val[p][1] == '@':
                    # escape escape
                    val[p] = after
                    continue
                # escape signifying type
                typemap = {"int":int, "float":float, "bool":bool, "str":str}
                if after in typemap:
                    val[p] = typemap[after]


    return val


    
        

def dict_to_str(params:dict, sigfigs:int=9, fmtchar='%') -> str:
    """
    Produces a GETlike stringified version of the (shallow) dictionary `params`
    e.g. {'a':4, 'b': 'string', 'c': True } -> "%a=4%b="string"%c=true

    @param params a dictionary of parameters to serialise
    @param sigfigs # significant figures to print for floats
    @param fmtchar 
    """
    s = ''
    for p in params:
        assert type(p) is str, f"non-string parameter {p} found in argument"
        s += f"{fmtchar}{p}="
        v = params[p]
        assert type(v) in [str, int, bool, float], f"Invalid parameter type: {v} (= {type(v)} )"
        if type(v) is float:
            s += stringly(v, sigfigs)
        elif type(v) is bool:
            s += 'true' if v else 'false'
        else:
            s += str(v)


def str_to_dict(fname:str, fmtchar="%", castdict=None):
    """
    Reads a GETlike string and parses it into a dict, ignoring any malformed pieces.
    @param fname string to be paarsed
    @param fmtchar data delineator
    @param castdict dictionary for casting variables 
    """
    params = {}
    
    label = path.basename(fname)
    tokens = label.split(fmtchar)
    for s in tokens:
        if s.count('=') == 1:
            # ignore any mangled tokens
            key, val = s.split('=')
            if castdict is None:
                params[key] = val
            elif callable(castdict[key]):
                params[key] = castdict[key](val)
            else:
                print(f"WARN: overriding [{key}] \t {castdict[key]} -> {val}")
                params[key] = type(castdict[key])(val)
                

    return params

def stringly(v, sigfigs:int=9):
    """
    @param v value to stringify
    @param sigfigs # sig figs to round to for value identififcation 
    """
    if type(v) is float:
        return ("{:."+str(sigfigs)+"e}").format(float(v))
    else:
        return v


class record(object):
    def __init__(self, filename:str, 
            schema: dict[str, int | float | str ], 
            data_load="csv") -> None:
        """
        @param filename name of the data file
        @param param_template: dict of parameters to use as default or look for in filename
        @param handler = [ csv | binary | custom f(filename, params) ] to turn file into ndarray
        @param ext File extension to ignore
        @param rules replacements
        """
        self.params = schema.copy()
        self.data = None
        self.set_loader(data_load)
        self.filepath = path.abspath(filename)
        assert path.isfile(self.filepath), f"Path '{filename}' is not a file"

    def set_params(self, overrides:dict):
        for k in overrides:
            assert k in self.params, f"Parameter '{k}' does not appear in the template."
            if self.params[k] in BASIC_TYPES:
                self.params[k] = self.params[k](overrides[k])
            else:
                # overriding a normal value
                print(f"WARN: overriding default value of {k}, {self.params[k]} -> {overrides[k]}")
                try:
                    self.params[k] = type(self.params[k])(overrides[k])
                except ValueError as e:
                    print(f"Failed to cast override {overrides[k]} to type {type(self.params[k])}.")
            
        
        
    def set_loader(self, data_load):
        # set the loader
        if callable(data_load):
            self.onload = data_load
        elif data_load=="csv":
            self.onload = lambda f, _ : np.genfromtxt(f, dtype=np.float64)
        elif data_load=="csv_names":
            self.onload = lambda f, _ : np.genfromtxt(f, dtype=np.float64, names=True)
        elif data_load == "binary_float":
            self.onload = lambda f, _ : np.fromfile(f, dtype=np.float64)
        elif data_load == "binary_cplx":
            self.onload = lambda f, _ : np.fromfile(f, dtype=np.complex128)
        else:
            raise ValueError('Handler is neither callable nor one of [ csv | csv_names | binary_float | binary_cplx ]')
        
    def get_data(self):
        if self.data is None:
            self.data = self.onload(self.filepath, self.params)
        return self.data
    
    def __str__(self):
        if self.data is None:
            return str(self.params) +"\nData: unloaded"
        else:
            return str(self.params) +f"\nData: {self.data.shape}, {self.data.dtype}"


class db_iterator(object):
    def __init__(self, parent):
        self._parent = parent
        self._idx_iterator = iter(parent.idx)

    def __next__(self):
        j = next(self._idx_iterator)
        return self._parent.records[j]



        
class db_view(object):
    def __init__(self, parent, idx:list) -> None:
        self.idx = idx # Reference to a list of indices
        self.records = parent.records
        self.schema = parent.schema
        self.atol = parent.atol
        self.rtol = parent.rtol
        self.rules = parent.rules

    def __iter__(self):
        return db_iterator(self)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i:int):
        return self.records[self.idx[i]]

    def where(self, param, value):
        '''
        Returns a db_view that contains only records with the appropriate infile parameter values
        @param param - the infile-param to select by
        @param value - the infile-param's value
        '''

        idx = []

        if param in self.schema:
            cast = self.schema[param]
        elif param in self.rules:
            # see if we can infer it
            cast = self.rules[param](self.records[0].params)
        else:
            raise KeyError(f"infile parameter [{param}] is undefined")
        
        if type(cast) in BASIC_TYPES:
            cast = type(cast)
        else:
            assert cast in BASIC_TYPES, f"Not possible to interpret {cast}"

        value = cast(value)

        for i in self.idx:
            r = self.records[i]
            if abs(r.params[param] - value) < self.atol or abs(r.params[param] - value)/(abs(value) + self.atol) < self.rtol:
                idx.append(i)

        if len(idx) == 0:
            raise IndexError("No records found.")

        return db_view(self, idx)

    def all(self):
        retval = []
        shape = np.shape(self.records[self.idx[0]].get_data())
        for i in self.idx:
            d = self.records[i].get_data()
            if (np.shape(d) != shape):
                raise RuntimeWarning(f"Record {i} has unexpected shape {np.shape(d)}\nDubious record params: {self.records[i].params}\n")
            retval.append(d)
        return retval


    def sum(self):
        s = self.records[self.idx[0]].get_data().copy()

        names = s.dtype.names
        if names is None:
            #  unstructured data
            for i in self.idx[1:]:
                r = self.records[i]
                s += r.get_data()
            
        else:
            for i in self.idx[1:]:
                r = self.records[i]
                new = r.get_data()
                for n in names:
                    s[n] += new[n]
            
        return s
    
    def sum2(self):
        s = self.records[self.idx[0]].get_data().copy()

        names = s.dtype.names
        if names is None:
            #  unstructured data
            for i in self.idx[1:]:
                r = self.records[i]
                s += r.get_data()**2
            
        else:
            for i in self.idx[1:]:
                r = self.records[i]
                new = r.get_data()**2
                for n in names:
                    s[n] += new[n]
            
            
        return s

    def mean(self):
        avg = self.sum()
        names = avg.dtype.names
        if names is None:
            avg /= len(self.idx)
        else:
            for n in names:
                avg[n] /= len(self.idx)
        return avg
    
    def var(self):
        avg = self.mean()
        avg2 = self.sum2()
        names = avg.dtype.names
        if names is None:
            avg2 /= len(self.idx)
        else:
            for n in names:
                avg2[n] /= len(self.idx)
        return avg2 - avg**2


    def __len__(self):
        return len(self.idx)
    
    @property
    def fixed_parameters(self):
        fixed, live = self.get_parameters()
        return fixed
        

    def get_parameters(self, sigfigs:int = None):
        if sigfigs is None:
            sigfigs = max(-round(np.log10(self.rtol)), 3)

        """
        Returns a dict of all parameters that are the same across the sample
        """
        common = self[0].params.copy()
        live = []
        for r in self:
            diff = []
            for c in common:
                if r.params[c] != common[c]:
                    diff.append(c)
            for d in diff:
                common.pop(d)
                live.append(d)

        return common, live

    @property
    def common_parameters(self, sigfigs:int=None):
        return self.get_parameters(sigfigs)[0]
    
    def common_parameter(self, key, sigfigs:int=None):
        const, vary = self.get_parameters(sigfigs)
        if key in const:
            return const[key]
        elif key in vary:
            raise KeyError(f"Key {key} is not constant in this sample.")
        elif key in self.schema:
            raise RuntimeError("Key neither varies nor is constant, there is a bug in db_view.get_parameters")
        else:
            raise KeyError(f"Key {key} is not in the parameter schema.")

    @property
    def live_parameters(self, sigfigs:int=None):
        retval = {}
        for p in self.get_parameters(sigfigs)[1]:
            retval[p] = list(self.unique_values(p).keys())
        return retval
        

    def unique_values(self, param:str, sigfigs:int=9):
        unique = {}
        for r in self:
            param_val = stringly(r.params[param], sigfigs)
            if param_val in unique:
                unique[param_val] += 1
            else:
                unique[param_val] = 1

        if param in self.schema:
            cast = self.schema[param]
        elif param in self.rules:
            # see if we can infer it
            cast = self.rules[param](self.records[0].params)
        else:
            raise ValueError(f"Parameter {param} is not knpwn to the database.")

        if cast not in BASIC_TYPES:
                cast = type(cast)

        retval = {}
        for k in unique:
            retval[cast(k)] = unique[k]

        return dict(sorted(retval.items()))
    


    
class db(db_view):
    def __init__(self, schema:dict[str, any], rtol=1e-9, atol=1e-9, rules = {}) -> None:
        """
        @param schema dict of any parameters, initialised to either default values or a type to signify coercion
        @param rtol relative tolerance to decide of floating params are the same
        @param atol absolute tolerance to decide of floating params are the same
        """
        self.records = []
        self.set_schema(schema)
        self.atol = atol
        self.rtol = rtol
        self.rules = rules

    def clear(self)->None:
        self.records = []
        self.set_schema({})

    def set_schema(self, schema:dict)->None:
        """
        Validates the schema
        
        """
        acceptable = [int, float, str, bool]
        if type(schema) is dict:
            for p in schema:
                assert (schema[p] in acceptable) or (type(schema[p]) in acceptable), f'illegal schema entry {schema[p]}'
            self.schema = schema.copy()
        else:
            self.schema = comprehend_parameters(schema)
            
    def set_rules(self, param_rules:dict):
        for k in param_rules:
            assert callable(param_rules[k]), f"Bad rule ({k}): must have form 'x': x(p:dict) -> schema.type(p)"
        self.rules = param_rules
           

    def add_record(self, filename:str, check_unique=True, data_load=None, ext=None) -> None:
        tmp =  record(filename, self.schema, data_load)
        if check_unique:
            for r in self.records:
                assert r.params != tmp.params, f"Attempting to add duplicate record "
        
        bn = path.basename(filename)
        # trim the extension
        bn = bn if ext is None else bn[:-len(ext)]
        # populate the running parameters using the rules provided
        tmp.set_params(str_to_dict(bn))

        # assert that all params are known
        if any([p in BASIC_TYPES for p in self.schema]):
            raise RuntimeError("Cannot apply rule: inadequate information.")
        # re-apply all rules
        for k in self.rules:
            try:
                tmp.params[k] = self.rules[k](tmp.params)
            except TypeError as e:
                print(e)
                print(f"Parameters for file {filename} are underdetermined")



        # ensure that all running parameters have been dealt with
        for k in tmp.params:
            assert type(tmp.params[k]) in BASIC_TYPES, f"Bad param: params[{k}] = {tmp.params[k]}"

        self.records.append(tmp)

    # def view(self):
    #     return db_view(self, self.idx)

    @property
    def idx(self) -> list:
        return list(range(len(self.records)))


    def load_files(self, datadir:str, startswith:str, endswith:str, data_load="csv"):
        for file in os.listdir(datadir):
            bn=path.basename(file)
            
            if bn.startswith(startswith) and bn.endswith(endswith):
                print(bn)
                self.add_record(os.path.join(datadir, file),
                    data_load=data_load,
                    ext = endswith
                    )



