import torch

from . import util


class Variable():
    def __init__(self, distribution=None, value=None, address_base=None, address=None, instance=None, log_prob=None, log_importance_weight=None, control=False, replace=False, name=None, observed=False, reused=False, tagged=False):
        self.distribution = distribution
        if value is None:
            self.value = None
        else:
            self.value = util.to_tensor(value)
        self.address_base = address_base
        self.address = address
        self.instance = instance
        if log_prob is None:
            self.log_prob = None
        else:
            self.log_prob = util.to_tensor(log_prob)
        if log_importance_weight is None:
            self.log_importance_weight = None
        else:
            self.log_importance_weight = float(log_importance_weight)
        self.control = control
        self.replace = replace
        self.name = name
        self.observable = ((not tagged) and (name is not None)) or observed
        self.observed = observed
        self.reused = reused
        self.tagged = tagged

    def __repr__(self):
        # The 'Unknown' cases below are for handling pruned variables in offline training datasets
        return 'Variable(name:{}, control:{}, replace:{}, observable:{}, observed:{}, tagged:{}, address:{}, distribution:{}, value:{}: log_prob:{})'.format(
            self.name if hasattr(self, 'name') else 'Unknown',
            self.control if hasattr(self, 'control') else 'Unknown',
            self.replace if hasattr(self, 'replace') else 'Unknown',
            self.observable if hasattr(self, 'observable') else 'Unknown',
            self.observed if hasattr(self, 'observed') else 'Unknown',
            self.tagged if hasattr(self, 'tagged') else 'Unknown',
            self.address if hasattr(self, 'address') else 'Unknown',
            str(self.distribution) if hasattr(self, 'distribution') else 'Unknown',
            str(self.value) if hasattr(self, 'value') else 'Unknown',
            str(self.log_prob) if hasattr(self, 'log_prob') else 'Unknown')

    def to(self, device):
        if self.value is not None:
            self.value.to(device=device)
        # if self.distribution is not None:
        #     self.distribution.to(device=device)

    def __hash__(self):
        return hash(self.address + str(self.value) + str(self.control) + str(self.replace) + str(self.observed) + str(self.tagged))

    def __eq__(self, other):
        return hash(self) == hash(other)


class Trace():
    def __init__(self):
        self.variables = []
        self.variables_controlled = []
        self.variables_uncontrolled = []
        self.variables_replaced = []
        self.variables_observed = []
        self.variables_observable = []
        self.variables_tagged = []
        self.variables_dict_address = {}
        self.variables_dict_address_base = {}
        self.named_variables = {}
        self.result = None
        self.log_prob = 0.
        self.log_prob_observed = 0.
        self.log_importance_weight = 0.
        self.length = 0
        self.length_controlled = 0
        self.execution_time_sec = None

    def __repr__(self):
        # The 'Unknown' cases below are for handling pruned traces in offline training datasets
        return 'Trace(all:{:,}, controlled:{:,}, replaced:{}, observeable:{}, observed:{}, tagged:{}, uncontrolled:{}, log_prob:{}, log_importance_weight:{})'.format(
            self.length,
            self.length_controlled,
            '{:,}'.format(len(self.variables_replaced)) if hasattr(self, 'variables_replaced') else 'Unknown',
            '{:,}'.format(len(self.variables_observed)) if hasattr(self, 'variables_observed') else 'Unknown',
            '{:,}'.format(len(self.variables_observable)) if hasattr(self, 'variables_observable') else 'Unknown',
            '{:,}'.format(len(self.variables_tagged)) if hasattr(self, 'variables_tagged') else 'Unknown',
            '{:,}'.format(len(self.variables_uncontrolled)) if hasattr(self, 'variables_uncontrolled') else 'Unknown',
            str(self.log_prob) if hasattr(self, 'log_prob') else 'Unknown',
            str(self.log_importance_weight) if hasattr(self, 'log_importance_weight') else 'Unknown')

    def add(self, variable):
        self.variables.append(variable)
        self.variables_dict_address[variable.address] = variable
        self.variables_dict_address_base[variable.address_base] = variable

    def end(self, result, execution_time_sec):
        self.result = result
        self.execution_time_sec = execution_time_sec
        replaced_indices = []
        for i in range(len(self.variables)):
            variable = self.variables[i]
            if variable.name is not None:
                self.named_variables[variable.name] = variable
            if variable.control and i not in replaced_indices:
                if variable.replace:
                    for j in range(i + 1, len(self.variables)):
                        if self.variables[j].address_base == variable.address_base:
                            self.variables_replaced.append(variable)
                            variable = self.variables[j]
                            replaced_indices.append(j)
                self.variables_controlled.append(variable)
        self.variables_uncontrolled = [v for v in self.variables if (not v.control) and (not v.observed) and (not v.tagged)]
        self.variables_observed = [v for v in self.variables if v.observed]
        self.variables_observable = [v for v in self.variables if v.observable]
        self.variables_tagged = [v for v in self.variables if v.tagged]
        self.log_prob = sum([torch.sum(v.log_prob) for v in self.variables if v.control or v.observed])
        self.log_prob_observed = sum([torch.sum(v.log_prob) for v in self.variables_observed])
        self.length = len(self.variables)
        self.length_controlled = len(self.variables_controlled)
        replaced_log_importance_weights = {}
        for variable in self.variables:
            if variable.log_importance_weight is not None:
                if variable.replace:
                    replaced_log_importance_weights[variable.address_base] = variable.log_importance_weight
                else:
                    self.log_importance_weight += variable.log_importance_weight
        for _, log_importance_weight in replaced_log_importance_weights.items():
            self.log_importance_weight += log_importance_weight

    def last_instance(self, address_base):
        if address_base in self.variables_dict_address_base:
            return self.variables_dict_address_base[address_base].instance
        else:
            return 0

    def to(self, device):
        for variable in self.variables:
            variable.to(device)

    def __hash__(self):
        h = [hash(variable) for variable in self.variables]
        return hash(sum(h))

    def __eq__(self, other):
        return hash(self) == hash(other)


class HDFLogger():

    def __init__(self, path, name, logging_struct, T_per_file=500000):
        name = "__".join(name.split("/")) # escape slash character in name
        self.logging_struct = logging_struct
        try:
            from tables import open_file
            import os
            self.path = path
            self.name = name
            self.T_per_file = T_per_file
            self.hdf_path = os.path.join(path, "hdf")
            self.folder_name =  os.path.join(self.hdf_path, name)
            if not os.path.isdir(self.folder_name):
                os.makedirs(self.folder_name)
        except Exception as e:
            self.logging_struct.console_logger.warning("Could not execute HDF logger save - no disk space, or no permissions? " +
                                                  "Error message: {}, path: {}, name: {}".format(e, path, name))

        pass

    def log(self, key, item, T_env):

        try:
            from tables import open_file, Filters
            file_T_id = T_env // self.T_per_file
            file_path = os.path.join(self.folder_name, "T_env_{}:{}.h5".format(file_T_id*self.T_per_file, (file_T_id + 1)*self.T_per_file))
            self.h5file = open_file(file_path,
                                    mode="a",
                                    title="Experiment results: {}".format(self.name))

            if isinstance(item, BatchEpisodeBuffer):

                    group = "learner_samples"+key
                    if not hasattr(self.h5file.root, group):
                        self.h5file.create_group("/", group, 'Learner samples')

                    if not hasattr(getattr(self.h5file.root, group), "T{}".format(T_env)):
                        self.h5file.create_group("/{}/".format(group), "T{}".format(T_env), 'Learner samples T_env:{}'.format(T_env))

                    if not hasattr(getattr(getattr(self.h5file.root, group), "T{}".format(T_env)), "_transition"):
                        self.h5file.create_group("/{}/T{}".format(group, T_env), "_transition", 'Transition-wide data')

                    if not hasattr(getattr(getattr(self.h5file.root, group), "T{}".format(T_env)), "_episode"):
                        self.h5file.create_group("/{}/T{}".format(group, T_env), "_episode", 'Episode-wide data')

                    filters = Filters(complevel=5, complib='blosc')

                    # if table layout has not been created yet, do it now:
                    for _c, _pos in item.columns._transition.items():
                        it = item.get_col(_c)[0].cpu().numpy()
                        if not hasattr(getattr(self.h5file.root, group), _c):
                            self.h5file.create_carray(getattr(getattr(self.h5file.root, group), "T{}".format(T_env))._transition,
                                                                    _c, obj=it, filters=filters)
                        else:
                            getattr(getattr(self.h5file.root, group)._transition, _c).append(it)
                            getattr(getattr(self.h5file.root, group)._transition, _c).flush()

                    # if table layout has not been created yet, do it now:
                    for _c, _pos in item.columns._episode.items():
                        it = item.get_col(_c, scope="episode")[0].cpu().numpy()
                        if not hasattr(getattr(self.h5file.root, group), _c):
                            self.h5file.create_carray(getattr(getattr(self.h5file.root, group), "T{}".format(T_env))._episode,
                                                                           _c, obj=it, filters=filters)
                        else:
                            getattr(getattr(self.h5file.root, group)._episode, _c).append(it)
                            getattr(getattr(self.h5file.root, group)._episode, _c).flush()

            else:

                key = "__".join(key.split(" "))
                # item needs to be scalar!#
                import torch as th
                import numpy as np
                if isinstance(item, th.Tensor):
                    item = np.array([item.cpu().clone().item()])
                elif not isinstance(item, np.ndarray):
                    item = np.array([item])

                if not hasattr(self.h5file.root, "log_values"):
                    self.h5file.create_group("/", "log_values", 'Log Values')

                if not hasattr(self.h5file.root.log_values, key):
                    from tables import Float32Atom, IntAtom
                    self.h5file.create_earray(self.h5file.root.log_values,
                                                                   key, atom=Float32Atom(), shape=[0])
                    self.h5file.create_earray(self.h5file.root.log_values,
                                                                   "{}_T_env".format(key), atom=IntAtom(), shape=[0])
                else:
                    getattr(self.h5file.root.log_values, key).append(item)
                    getattr(self.h5file.root.log_values, key).flush()

                    getattr(self.h5file.root.log_values, "{}_T_env".format(key)).append(np.array([T_env]))
                    getattr(self.h5file.root.log_values, "{}_T_env".format(key)).flush()


            self.h5file.close()

        except Exception as e:
            self.logging_struct.console_logger.warning("Could not execute HDF logger save - no disk space, or no permissions? Error message: {}, T_env: {}, key: {}, item: {}".format(e, T_env, key, str(item)))

        return


class TraceHash():
    """
    This storage class is designed for situations where the amount of trace data produced during a
    single forward() call is too large to be stored in RAM without compression.

    TraceHash uses a 2-stage compression process:


        First, each trace is hashed using a nested approach. Associative access is preserved.
        For EMOD, this alone yields a factor 22 memory compression with very little overhead.
        To do this: A dictionary over function names is built and updated online.
                    Each trace added is decomposed into individual function names and each function name is replaced by the
                    corresponding counter from the dictionary.
            Set :trace_hash_byte_len: to determine how many bytes
                (2 is fine for EMOD, you probably never need more than 4 even for very large code
                bases)

        Secondly, each hashed trace is compressed using a general purpose compression algorithm.
            Set :compression_mode: to "blosc"
            Set :file_path: to determine the path under which the trace file is stored. If set to <None>,
                traces are not stored on disk but stay in RAM.
                NOTE: The traces generated are written to disk in chunks determined by :chunk_size: (byte)

    """
    def __init__(self, file_name):
        self.variables = []
        self.variables_controlled = []
        self.variables_uncontrolled = []
        self.variables_replaced = []
        self.variables_observed = []
        self.variables_observable = []
        self.variables_tagged = []
        self.variables_dict_address = {}
        self.variables_dict_address_base = {}
        self.named_variables = {}
        self.result = None
        self.log_prob = 0.
        self.log_prob_observed = 0.
        self.log_importance_weight = 0.
        self.length = 0
        self.length_controlled = 0
        self.execution_time_sec = None

        self.trace_hash_byte_len = 1 # fixed for now
        self.funcname_lookup = {} # contains the lookup table for the function names
        self.file_name = file_name

    def _hash_funcname(self, variable):
        base, app = variable.address[1:].split("]")
        _tmp = app.split("__")
        base2 = "__".join(_tmp[:-1])
        new_address_base = _tmp[-1]
        base = base.split("; ") + [base2]
        new_address = b""
        for b in base:
            if b not in self.funcname_lookup:
                self.funcname_lookup[b] = (len(self.funcname_lookup) + 1).to_bytes(self.trace_hash_byte_len, "little") # avoid 0 as key
            new_address += self.funcname_lookup[b]
        return new_address, new_address_base

    def __repr__(self):
        return 'Trace(all:{:,}, controlled:{:,}, replaced:{:,}, observeable:{:,}, observed:{:,}, tagged:{:,}, uncontrolled:{:,}, log_prob:{}, log_importance_weight:{})'.format(
            len(self.variables),
            len(self.variables_controlled),
            len(self.variables_replaced),
            len(self.variables_observable),
            len(self.variables_observed),
            len(self.variables_tagged),
            len(self.variables_uncontrolled),
            str(self.log_prob),
            str(self.log_importance_weight))

    def add(self, variable):
        new_address, new_address_base = self._hash_funcname(variable)
        variable.address = new_address + bytes((new_address_base).encode("ascii"))
        variable.address_base = new_address
        variable.hash_funcname = self._hash_funcname
        self.variables.append(variable)
        self.variables_dict_address[variable.address] = variable
        self.variables_dict_address_base[variable.address_base] = variable

    def end(self, result, execution_time_sec):
        self.result = result
        self.execution_time_sec = execution_time_sec
        replaced_indices = []
        for i in range(len(self.variables)):
            variable = self.variables[i]
            if variable.name is not None:
                self.named_variables[variable.name] = variable
            if variable.control and i not in replaced_indices:
                if variable.replace:
                    for j in range(i + 1, len(self.variables)):
                        if self.variables[j].address_base == variable.address_base:
                            self.variables_replaced.append(variable)
                            variable = self.variables[j]
                            replaced_indices.append(j)
                self.variables_controlled.append(variable)
        self.variables_uncontrolled = [v for v in self.variables if (not v.control) and (not v.observed) and (not v.tagged)]
        self.variables_observed = [v for v in self.variables if v.observed]
        self.variables_observable = [v for v in self.variables if v.observable]
        self.variables_tagged = [v for v in self.variables if v.tagged]
        self.log_prob = sum([torch.sum(v.log_prob) for v in self.variables if v.control or v.observed])
        self.log_prob_observed = sum([torch.sum(v.log_prob) for v in self.variables_observed])
        self.length = len(self.variables)
        self.length_controlled = len(self.variables_controlled)
        replaced_log_importance_weights = {}
        for variable in self.variables:
            if variable.log_importance_weight is not None:
                if variable.replace:
                    replaced_log_importance_weights[variable.address_base] = variable.log_importance_weight
                else:
                    self.log_importance_weight += variable.log_importance_weight
        for _, log_importance_weight in replaced_log_importance_weights.items():
            self.log_importance_weight += log_importance_weight

    def last_instance(self, address_base):
        if address_base in self.variables_dict_address_base:
            return self.variables_dict_address_base[address_base].instance
        else:
            return 0

    def to(self, device):
        for variable in self.variables:
            variable.to(device)

    def __hash__(self):
        h = [hash(variable) for variable in self.variables]
        return hash(sum(h))

    def __eq__(self, other):
        return hash(self) == hash(other)

# import shelve
#
# class TraceShelve():
#     """
#     Implemented by Christian Schroeder de Witt April 2019
#
#     """
#     def __init__(self, file_name, file_sync_timeout=100):
#
#         shelf_flag = "n"
#         self._file_name = file_name
#         self._shelf = shelve.open(self._file_name,
#                                   flag=shelf_flag,
#                                   writeback=False)
#
#         self._shelf["variables"] = []
#         self._shelf["variables_controlled"] = []
#         self._shelf["variables_uncontrolled"] = []
#         self._shelf["variables_replaced"] = []
#         self._shelf["variables_observed"] = []
#         self._shelf["variables_observable"] = []
#         self._shelf["variables_tagged"] = []
#         self._shelf["variables_dict_address"] = {}
#         self._shelf["variables_dict_address_base"] = {}
#         self._shelf["named_variables"] = {}
#         self.variables = self._shelf["variables"]
#         self.variables_controlled = self._shelf["variables_controlled"]
#         self.variables_uncontrolled = self._shelf["variables_uncontrolled"]
#         self.variables_replaced = self._shelf["variables_replaced"]
#         self.variables_observed = self._shelf["variables_observed"]
#         self.variables_observable = self._shelf["variables_observable"]
#         self.variables_tagged = self._shelf["variables_tagged"]
#         self.variables_dict_address = self._shelf["variables_dict_address"]
#         self.variables_dict_address_base = self._shelf["variables_dict_address_base"]
#         self.named_variables = self._shelf["named_variables"]
#
#         self.result = None
#         self.log_prob = 0.
#         self.log_prob_observed = 0.
#         self.log_importance_weight = 0.
#         self.length = 0
#         self.length_controlled = 0
#         self.execution_time_sec = None
#
#         self._file_sync_countdown = 0
#         self.file_sync_timeout = file_sync_timeout
#
#         self.entry_ctr = 0
#
#     def __repr__(self):
#         # The 'Unknown' cases below are for handling pruned traces in offline training datasets
#         return 'Trace(all:{:,}, controlled:{:,}, replaced:{}, observeable:{}, observed:{}, tagged:{}, uncontrolled:{}, log_prob:{}, log_importance_weight:{})'.format(
#             self.length,
#             self.length_controlled,
#             '{:,}'.format(len(self._shelf["variables_replaced"])) if 'variables_replaced' in self._shelf else 'Unknown',
#             '{:,}'.format(len(self._shelf["variables_observed"])) if 'variables_observed' in self._shelf else 'Unknown',
#             '{:,}'.format(len(self._shelf["variables_observable"])) if 'variables_observable' in self._shelf else 'Unknown',
#             '{:,}'.format(len(self._shelf["variables_tagged"])) if 'variables_tagged' in self._shelf else 'Unknown',
#             '{:,}'.format(len(self._shelf["variables_uncontrolled"])) if 'variables_uncontrolled' in self._shelf else 'Unknown',
#             str(self.log_prob) if hasattr(self, 'log_prob') else 'Unknown',
#             str(self.log_importance_weight) if hasattr(self, 'log_importance_weight') else 'Unknown')
#
#     def add(self, variable):
#         self._shelf["variables__{}".format(self.entry_ctr)] = variable
#         # tmp = self._shelf["variables"]
#         # tmp.append(variable)
#         # self._shelf["variables"] = tmp
#         tmp = self._shelf["variables_dict_address"]
#         tmp[variable.address] = variable
#         self._shelf["variables_dict_address"] = tmp
#         tmp = self._shelf["variables_dict_address_base"]
#         tmp[variable.address_base] = variable
#         self._shelf["variables_dict_address_base"] = tmp
#         self._file_sync_countdown += 1
#         if self._file_sync_countdown >= self.file_sync_timeout:
#             print("SYNCING TRACE SHELF")
#             self._shelf.sync()
#             self._file_sync_countdown = 0
#
#         self.entry_ctr += 1
#
#     def end(self, result, execution_time_sec):
#         self.result = result
#         self.execution_time_sec = execution_time_sec
#         replaced_indices = []
#         for i in range(len(self._shelf["variables"])):
#             variable = self._shelf["variables"][i]
#             if variable.name is not None:
#                 self._shelf["named_variables"][variable.name] = variable
#             if variable.control and i not in replaced_indices:
#                 if variable.replace:
#                     for j in range(i + 1, len(self._shelf["variables"])):
#                         if self._shelf["variables"][j].address_base == variable.address_base:
#                             self._shelf["variables_replaced"].append(variable)
#                             variable = self._shelf["variables"][j]
#                             replaced_indices.append(j)
#                 self._shelf["variables_controlled"].append(variable)
#         self._shelf["variables_uncontrolled"] = [v for v in self._shelf["variables"] if (not v.control) and (not v.observed) and (not v.tagged)]
#         self._shelf["variables_observed"] = [v for v in self._shelf["variables"] if v.observed]
#         self._shelf["variables_observable"] = [v for v in self._shelf["variables"] if v.observable]
#         self._shelf["variables_tagged"] = [v for v in self._shelf["variables"] if v.tagged]
#         self.log_prob = sum([torch.sum(v.log_prob) for v in self._shelf["variables"] if v.control or v.observed])
#         self.log_prob_observed = sum([torch.sum(v.log_prob) for v in self._shelf["variables_observed"]])
#         self.length = len(self._shelf["variables"])
#         self.length_controlled = len(self._shelf["variables_controlled"])
#         replaced_log_importance_weights = {}
#         for variable in self._shelf["variables"]:
#             if variable.log_importance_weight is not None:
#                 if variable.replace:
#                     replaced_log_importance_weights[variable.address_base] = variable.log_importance_weight
#                 else:
#                     self.log_importance_weight += variable.log_importance_weight
#         for _, log_importance_weight in replaced_log_importance_weights.items():
#             self.log_importance_weight += log_importance_weight
#         self._shelf.close()
#
#     def last_instance(self, address_base):
#         if address_base in self._shelf["variables_dict_address_base"]:
#             return self._shelf["variables_dict_address_base"][address_base].instance
#         else:
#             return 0
#
#     def to(self, device):
#         for variable in self._shelf["variables"]:
#             variable.to(device)
#
#     def __hash__(self):
#         h = [hash(variable) for variable in self._shelf["variables"]]
#         return hash(sum(h))
#
#     def __eq__(self, other):
#         return hash(self) == hash(other)
