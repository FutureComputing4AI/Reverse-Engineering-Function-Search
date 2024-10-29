import multiprocessing
import multiprocessing.dummy
import os
from pathlib import Path
from abc import ABC, abstractmethod
import shutil
from tempfile import TemporaryDirectory
from time import time
import typing
from tqdm import tqdm
from hashlib import md5

import executable_filter as filters


class GhidraMultithreader:

    def __init__(self,
                 binaries: str | bytes | os.PathLike | typing.List[str | bytes | os.PathLike],
                 ghidra_support_dir: str | bytes | os.PathLike,
                 progress_bar_description: str,
                 verification: typing.Callable,
                 skip_existing: bool = True,
                 filter: filters.ExecutableFilter = None,
                 logging_dir: str | bytes | os.PathLike = "./logs/",
                 ghidra_tmp_dir: str | bytes | os.PathLike = "/tmp",
                 max_attempts: int = 3):
        
        ### Establish Binaries Home and master-list of files

        self.filter = filter or (lambda _ : True)

        if isinstance(binaries, (list, set)):
            # self._get_master_list_from_list_of_paths(binaries)
            self.master_list = list(binaries)
        elif isinstance(binaries , (str, bytes, os.PathLike)):
            self._get_master_list_from_directory(binaries)
        else:
            raise ValueError(f"List of binaries must be str | bytes | os.PathLike, or List/Set of those. Got {type(binaries)}")

        ### Establish Ghidra scripts exist
        ghidra_home = Path(ghidra_support_dir)

        headless_script_path = ghidra_home / "analyzeHeadless"
        if not (headless_script_path).exists():
            raise FileNotFoundError(f"Cannot find Ghidra headless analyzer: {headless_script_path}")

        self.ghidra_home = ghidra_home
        self.headless_script_path = headless_script_path

        ### Logging
        self.logging_dir = Path(logging_dir)
        self.logging_dir.mkdir(parents=True, exist_ok=True)

        self.prune_logs = True

        ### Temporary Project Storage
        self.ghidra_tmp_dir = Path(ghidra_tmp_dir)
        self.ghidra_tmp_dir.mkdir(parents=True, exist_ok=True)

        ### Max Attempts
        if max_attempts < 0:
            raise ValueError(f"Max Attempts must be postive integer, got {max_attempts}")
        self.max_attempts = max_attempts

        ### Verification
        self.verification = verification
        self.skip_existing_path = skip_existing

        ### Everything Else

        self.dry_run = False

        self.prune_logs = True

        self.pbar = None
        self.description = progress_bar_description




    def _get_master_list_from_list_of_paths(self, list_of_binaries: typing.List[str | bytes | os.PathLike]):
        if len(list_of_binaries) == 0:
            raise ValueError("List of files must be non-empty")

        self.master_list = []
        for path in list_of_binaries:
            # path = Path(path)

            if not os.path.exists(path):
                raise FileNotFoundError(path)
            
            if os.path.isdir(path):
                raise ValueError(f"List of paths must be only files, got directory {path}")
            
            if self.filter(path):
                self.master_list.append(path)


    def _get_master_list_from_directory(self, binaries_path: str | bytes | os.PathLike):
        binaries_path = Path(binaries_path)
        if not binaries_path.exists():
            raise FileNotFoundError(binaries_path)
        
        if not binaries_path.is_dir():
            raise ValueError(f"Path to binaries must be directory, or a list of files, got {binaries_path}")
        
        self.master_list = [str(path) for path in binaries_path.rglob("*") if path.is_file() and self.filter(path)]


    @abstractmethod
    def _process_one(self, path: str | bytes | os.PathLike) -> None:
        """Process one Binary from its path

        Args:
            path (str | bytes | os.PathLike): path to binary
        """        
        return

    @staticmethod
    def _noncompliant(path: str | bytes | os.PathLike) -> bool:
        """Returns `True` if ghidra will not be able to elegantly process the path to a binary, 
        which can happen if the path:
          - Contains spaces
          - Contains non-ascii characters
          - Is a symlink

        Parameters
        ----------
        path : str | bytes | os.PathLike
            Path to the binary
        """        
        path = str(path)

        return (not path.isascii()) or \
               (" " in path) or \
               (Path(path).is_symlink())

    def _check_and_fix_noncompliant_path(self, 
                                     path: str | bytes | os.PathLike, 
                                     project_dir: str | bytes | os.PathLike
                                     ) -> typing.Optional[Path]:
        path, new_path = Path(path), None

        if GhidraMultithreader._noncompliant(path):
            new_path = Path(project_dir) / f"{md5(open(path, 'rb').read()).hexdigest()}"
            old_pdb = path.with_suffix(".pdb")
            
            shutil.copy(path, new_path)

            if old_pdb.exists(): 
                shutil.copy(path, new_path.with_suffix(".pdb"))

        return new_path         


    @abstractmethod
    def _call(self, 
              path: str | bytes | os.PathLike, 
              project_dir: str | bytes | os.PathLike, 
              logger: typing.TextIO):
        raise NotImplementedError
    

    def _process_one(self, path: str | bytes | os.PathLike):

        try:
            ### Skip existing
            if self.skip_existing_path and self.verification(path):
                self.pbar.update()
                return
            
            individual_logging_dir = self.logging_dir / path.replace("/", '-').replace('.', '\\')
            individual_logging_dir.mkdir(parents=True, exist_ok=True)

            ### Processing
            for attempt_num in range(1, self.max_attempts + 1):
                with open(individual_logging_dir / f"Attempt_{attempt_num}.txt", 'w+') as logging_file, \
                TemporaryDirectory(dir=self.ghidra_tmp_dir) as project_dir:
                    
                    ### Copy the file into the project directory if the path would break Ghidra Imports
                    ### This could happen to symlinks, or paths with spaces/non-ascii characters
                    new_path = self._check_and_fix_noncompliant_path(path, project_dir)
                    if new_path is not None:
                        old_path = path
                        path = new_path
                        # self.pbar.write(f"Non-compliant path: {old_path}")
                    
                    self._call(path, project_dir, logging_file)

                    if new_path is not None: path = old_path ### Change back for post_verification

                ### Verification

                if self.verification(path):
                    if self.prune_logs:
                        shutil.rmtree(individual_logging_dir)

                    self.pbar.update()

                    return
        except Exception as e:
            self.pbar.write(f"File {path} raise exception: \n{str(e)}")

        self.pbar.write(f"Process Failed for {path} - unable to verify success after three attempts")


    def run(self, num_workers: int = 2, limit: int = None) -> None:
        start = time()

        process = self.master_list
        if limit is not None:
            if limit < 0: raise ValueError(f"Limit must be greater than 0, got {limit}")
            process = process[:limit]

        self.num_workers = min(len(process), num_workers)
        N = len(process)
        self.pbar = tqdm(total=N)
        self.pbar.set_description(self.description)

        pool = multiprocessing.dummy.Pool(self.num_workers)
        self.pbar.write(f"Initialized multiprocessing pool with {self.num_workers} workers.")
        
        if self.prune_logs:
            self.pbar.write("Log pruning enabled.")

        pool.map(self._process_one, process)

        end = time()

        self.pbar.write(
            f"Processed {len(process)} in {end - start :.2f}s. {(end - start) / len(process):.4f}s per binary"
        )
