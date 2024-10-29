import subprocess
from dotenv import load_dotenv
import os
from pathlib import Path
from time import time
from hashlib import sha256
from argparse import ArgumentParser

import executable_filter as filters
from signature_generator import SignatureGenerator
from postscript_multithreader import PostScriptMultithreader
from MRR import analysis


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("binaries")
    args = parser.parse_args()

    load_dotenv()

    MAKE_SIGS, COMMIT, QUERY, MRR = 1,1,1,1
    MOTIF_MODE = False

    log_dir = Path(os.getenv("LOGGING_DIR", "./logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    with open(log_dir / f'timings_{time()}.txt', 'a+') as time_log:

        # ghidra_support_dir = Path("/home/nkfleis/ghidra/support/")
        ghidra_home = os.getenv("GHIDRA_HOME")
        if ghidra_home is None:
            raise EnvironmentError("Specifiy GHIDRA_HOME in environment or in .env")
        ghidra_support_dir = Path(ghidra_home) / "support"

        binaries = Path(args.binaries)
        if binaries.suffix == ".txt":
            binaries = filters.FileFilter(binaries).fileset
            file_filter = None
        else:
            file_filter = filters.ExcludePDBs

        output_dir = os.getenv("OUTPUT_HOME")
        if output_dir is None:
            raise EnvironmentError("Specify OUTPUT_HOME to save signatures, queries, and temporary_projects")
        output_dir = Path(output_dir)

        signature_path =  output_dir / "signatures"
        query_output_dir = output_dir / "bsim_queries"
        ghidra_tmp_dir = output_dir / "projects"

        ghidra_tmp_dir.mkdir(parents=True, exist_ok=True)

        ### Create Signatures
        if MAKE_SIGS:
            signature_path.mkdir(parents=True, exist_ok=True)

            generator = SignatureGenerator(
                binaries=binaries,
                ghidra_support_dir=ghidra_support_dir,
                signature_save_dir=signature_path,
                filter=file_filter,
                logging_dir=log_dir / "sigs",
                ghidra_tmp_dir=ghidra_tmp_dir
            )

            sig_start = time()

            generator.run(os.cpu_count() - 1)

            time_log.write(f"Generated {len(os.listdir(signature_path))} sigs in {time() - sig_start:.2f} seconds")

        ### Commit Signatures
        if COMMIT:
            commit_start = time()

            subprocess.run(
                f"{ghidra_support_dir / 'bsim'} commitsigs {os.getenv('BSIM_URL')} {signature_path}".split(),
                input=bytes(os.getenv("BSIM_PASSWORD"), encoding='utf-8')
            )

            time_log.write(f"Committed in {time() - commit_start :.2f}s")

        

        # ### Run Query Script
        if QUERY:
            query_output_dir.mkdir(parents=True, exist_ok=True)

            def verify_query(path: str | bytes | os.PathLike):


                exec_sha256 = sha256(open(path, 'rb').read()).hexdigest()
                return (query_output_dir / exec_sha256[:3]).exists() and \
                        f"{exec_sha256}" in os.listdir(query_output_dir / exec_sha256[:3])

            ghidra_script_name = "MOTIFBSimQuery" if MOTIF_MODE else "SingleFileBSimQuery"

            mt_script = PostScriptMultithreader(
                binaries=binaries,
                ghidra_support_dir=ghidra_support_dir,
                ghidra_script_name=ghidra_script_name,
                ghidra_script_path=os.getenv('SCRIPT_DIR'),
                ghidra_script_args=[str(query_output_dir), os.getenv('BSIM_URL')],
                logging_dir=log_dir / "query",
                postscript_verification=verify_query,
                ghidra_tmp_dir=ghidra_tmp_dir,
                analysis_timeout=600,
                filter=file_filter
            )

            analysis_start = time()

            mt_script.run(os.cpu_count() - 1)

            time_log.write(f"Completed queries in {time() - analysis_start:.2f}s")

        ### Calculate MRR
        if MRR:
            mrr_start_time = time()

            motif_metadata_filepath = os.getenv("MOTIF_METADATA_PATH") if MOTIF_MODE else None
            upper_mrr, lower_mrr = analysis(query_output_dir, motif_metadata_filepath=motif_metadata_filepath)

            time_log.write(f"Completed MRR in {time() - mrr_start_time:.2f}s.\n Upper: {upper_mrr} Lower {lower_mrr}")






    



