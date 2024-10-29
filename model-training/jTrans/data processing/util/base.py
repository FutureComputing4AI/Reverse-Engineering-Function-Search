from elftools.elf.elffile import ELFFile
import elftools.elf.elffile as elffile
from elftools.elf.sections import SymbolTableSection
from collections import defaultdict
import os
import json
import sys
import os
import sqlite3
import glob
from tqdm import tqdm
import shutil
import pefile


dbfile = "sept25.sqlite"

class Binarybase(object):
    def __init__(self, unstrip_path):
        self.unstrip_path = unstrip_path
        self.addr2name = self.extract_addr2name(self.unstrip_path)

    def get_func_name(self, name, functions):
        if name not in functions:
            return name
        i = 0
        while True:
            new_name = name+'_'+str(i)
            if new_name not in functions:
                return new_name
            i += 1

    def scan_section(self, functions, section):
        """
        Function to extract function names from a shared library file.
        """
        if not section or not isinstance(section, SymbolTableSection) or section['sh_entsize'] == 0:
            return 0

        count = 0
        for nsym, symbol in enumerate(section.iter_symbols()):

            if symbol['st_info']['type'] == 'STT_FUNC' and symbol['st_shndx'] != 'SHN_UNDEF':

                func = symbol.name

                name = self.get_func_name(func, functions)

                if not name in functions:

                    functions[name] = {}

                functions[name]['begin'] = symbol.entry['st_value']


    def extract_addr2name(self, path):
        '''
        return:
        '''
        
        basepath = os.path.dirname(path)
        basename = os.path.basename(path)
        binid = basepath.split("/")[-1]

        addr2name = {}

        with open(f'Assemblage_data/addr_ref/{binid}.json', 'r') as f:
            db = json.load(f)
        peobj =  pefile.PE(path, fast_load=True)

        for name in db:
            addr2name[peobj.get_physical_by_rva(db[name])] = name
        
        return addr2name
