import subprocess
from pathlib import Path
import os
from typing import List
import typing
from dotenv import load_dotenv
from hashlib import md5

import executable_filter as filters
from ghidra_multithreader import GhidraMultithreader



class SignatureGenerator(GhidraMultithreader):
    
    def __init__(self, 
                 binaries: str | bytes | os.PathLike | List[str | bytes | os.PathLike], 
                 ghidra_support_dir: str | bytes | os.PathLike,
                 signature_save_dir: str | bytes | os.PathLike,
                 filter: filters.ExecutableFilter = None,
                 logging_dir: str | bytes | os.PathLike = "./logs/",
                 ghidra_tmp_dir: str | bytes | os.PathLike = "/tmp",
                 max_attempts: int = 3):
        """_summary_

        Parameters
        ----------
        binaries (str | bytes | os.PathLike | List[str | bytes | os.PathLike]): 
            Parent directory of binaries to search through and process recursively or a list of
            files to process non-recursively.
        ghidra_support_dir : str | bytes | os.PathLike
            Directory containing `analyzeHeadles` and `bsim` scripts
        signature_save_dir : str | bytes | os.PathLike
            Output directory for signatures
        filter : filters.ExecutableFilter, optional
            Filter for files in binaries directory, by default None
        logging_dir : str | bytes | os.PathLike, optional
            Directory to keep logs (by default, logs are pruned if the signature is generated
            successfully), by default "./logs/"
        ghidra_tmp_dir : str | bytes | os.PathLike, optional
            Directory to store temporarily store ghidra projects, by default "/tmp"
        max_attempts : int, optional
            maximum number of time to attempt signature generation, by default 3
        """             
        super().__init__(
            binaries=binaries, 
            ghidra_support_dir=ghidra_support_dir, 
            progress_bar_description="Creating Signatures",
            verification=self.verify_signature,
            skip_existing=True,
            filter=filter,
            logging_dir=logging_dir,
            ghidra_tmp_dir= ghidra_tmp_dir,
            max_attempts=max_attempts)

        ### Verify BSIM interface script
        bsim_script_path = self.ghidra_home / 'bsim'
        if not (bsim_script_path).exists():
            raise FileNotFoundError(f"Cannot find Ghidra bsim script: {bsim_script_path}")

        ### Establish save dir for signatures
        self.signature_save_dir = Path(signature_save_dir)
        self.signature_save_dir.mkdir(parents=True, exist_ok=True)

        ### Everything else
        load_dotenv()

        self.import_command = f"{self.headless_script_path} {{}} Hydra -import {{}}"
        self.signature_command = f"{bsim_script_path} generatesigs ghidra:{{}}/Hydra {self.signature_save_dir} --bsim {os.getenv('BSIM_URL')}"


    def verify_signature(self, path: str | bytes | os.PathLike):
        exec_md5 = md5(open(path, 'rb').read()).hexdigest()
        return f"sigs_{exec_md5}" in os.listdir(self.signature_save_dir)


    def _call(self, 
              path: str | bytes | os.PathLike, 
              project_dir: str | bytes | os.PathLike, 
              logger: typing.TextIO):

        subprocess.run(
            self.import_command.format(project_dir, path).split(),
            stdout=logger,
            stderr=logger
        )

        subprocess.run(
            self.signature_command.format(project_dir).split(),
            stdout=logger,
            stderr=logger,
            input=bytes(os.getenv('BSIM_PASSWORD'), encoding='utf-8')
        )
