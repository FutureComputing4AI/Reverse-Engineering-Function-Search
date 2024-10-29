import os
import subprocess
from typing import Callable, List
import typing

import executable_filter as filters 
from ghidra_multithreader import GhidraMultithreader


class PostScriptMultithreader(GhidraMultithreader):

    def __init__(self, 
                 binaries: str | bytes | os.PathLike | List[str | bytes | os.PathLike], 
                 ghidra_support_dir: str | bytes | os.PathLike, 
                 ghidra_script_name: str,
                 ghidra_script_path: str | bytes | os.PathLike,
                 ghidra_script_args: List[str] = None,
                 postscript_verification: Callable = None,
                 filter: filters.ExecutableFilter = None,
                 logging_dir: str | bytes | os.PathLike = "./logs/",
                 analysis_timeout: int = None,
                 ghidra_tmp_dir: str | bytes | os.PathLike = "/tmp",
                 max_attempts: int = 3):
        """PostScriptMultithreader parallelizes the process of importing a file into a new ghidra
        project, running a GhidraScript, and then deleting the project. This is useful for splitting
        an executable into individual functions, or querying the executable against a bsim server.

        Args:
            binaries (str | bytes | os.PathLike | List[str | bytes | os.PathLike]): 
                Parent directory of binaries to search through and process recursively or a list of
                files to process non-recursively.
            ghidra_support_dir (str | bytes | os.PathLike): 
                Directory containing `analyzeHeadless` script
            ghidra_script_name (str): 
                The name of the ghidra script/java class to call
            ghidra_script_path (str | bytes | os.PathLike):
                Path to directory containing the above file
            ghidra_script_args (List[str]):
                List of script args to pass to the scipt, such as a BSim database URL or output 
                directory. Defaults to None
            postscript_verification (Callable):
                A function to verify successful script completion, called after process completes. 
                If false, another attempt will be triggered if available.
                To skip: use `None`
            filter (filters.ExecutableFilter, optional):
                Filter for files in binaries directory. Defaults to None.
            logging_dir : str | bytes | os.PathLike, optional
                Directory to keep logs (by default, logs are pruned if the signature is generated
                successfully), by default "./logs/"
            analysis_timeout (int, optional): 
                Ghidra import analysis timeout. `None` for no limit. Defaults to None.
            ghidra_tmp_dir : str | bytes | os.PathLike, optional
                Directory to store temporarily store ghidra projects, by default "/tmp"
            max_attempts : int, optional
                maximum number of time to attempt signature generation, by default 3
        """        
        super().__init__(
            binaries, 
            ghidra_support_dir, 
            progress_bar_description=f"Running Script - {ghidra_script_name}", 
            verification = postscript_verification or (lambda _ : True),
            skip_existing = (postscript_verification is not None),
            filter=filter,
            logging_dir=logging_dir,
            ghidra_tmp_dir=ghidra_tmp_dir,
            max_attempts=max_attempts)
        
        ### Everything else
        script_args = " ".join(ghidra_script_args or [])
        self.command = f"{self.headless_script_path} {{}} Hydra -import {{}} -postScript {ghidra_script_name} {script_args} -scriptPath {ghidra_script_path} -deleteProject"

        self.postscript_verification = postscript_verification

        self.prune_logs = True

        if analysis_timeout is not None:
            assert analysis_timeout > 0, f"Analysis timeout must be None or PosInt. Got {analysis_timeout}"
            self.command = f"{self.command} -analysisTimeoutPerFile {analysis_timeout}"

    
    def _call(self, 
              path: str | bytes | os.PathLike, 
              project_dir: str | bytes | os.PathLike, 
              logger: typing.TextIO):

        subprocess.call(
            self.command.format(project_dir, path).split(),
            stdout=logger,
            stderr=logger
        )
