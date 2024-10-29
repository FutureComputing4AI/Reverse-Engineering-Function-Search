from abc import ABC, abstractmethod
import os
import re
from typing import List
import numpy as np

class ExecutableFilter(ABC):
    
    @abstractmethod
    def __call__(self, path: str | bytes | os.PathLike) -> bool:
        pass
    

class RegexFilter(ExecutableFilter):

    def __init__(self, regex: str | bytes, exclude_mode=False):
        """Filters paths that pass/fail the given regex (changed according to exclude_mode, set to
        true to exclude paths that pass)

        Parameters
        ----------
        regex : str | bytes
            Regular expression filter
        exclude_mode : bool, optional
            If true, exclude paths that pass the regex, by default False
        """

        self.filter = re.compile(regex)
        self.exclude_mode = exclude_mode    


    def __call__(self, path: str | bytes | os.PathLike) -> bool:
        """
        
        +-------------------+------+-------+
        | Match is not None | Mode | Allow |
        |==================================|
        | T                 | T    | No    |
        | T                 | F    | Yes   |
        | F                 | T    | Yes   |
        | F                 | F    | No    |
        +-------------------+------+-------+

        Parameters
        ----------
        path : str | bytes | os.PathLike
            _description_

        Returns
        -------
        bool
            _description_
        """        
        return np.logical_xor(self.filter.search(str(path)) is not None, self.exclude_mode)
    


class FileFilter(ExecutableFilter):

    def __init__(self, list_file: str | bytes | os.PathLike, exclude_mode=False):
        
        self.fileset = set()
        with open(list_file, 'r') as infile:
            
            for line in infile.readlines():
                line = line.strip()
                self.fileset.add(line)

        self.exclude_mode = exclude_mode


    def __call__(self, path: str | bytes | os.PathLike) -> bool:
        if self.exclude_mode:
            # In set     -> Exclude
            # Not in set -> Include
            return str(path) not in self.fileset

        # In set     -> Include
        # Not in set -> Exclude        
        return str(path) in self.fileset


class FilterChain(ExecutableFilter):

    def __init__(self, *filters: List[ExecutableFilter]):
        self.filters = filters

    def __call__(self, path: str | bytes | os.PathLike) -> bool:
        return np.all([f(path) for f in self.filters])

ExcludePDBs = RegexFilter(".+\.pdb", exclude_mode=True)
OnlyEXEs = RegexFilter(".+\.exe")
