#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##############################################################################
#                                                                            #
#  Code for the USENIX Security '22 paper:                                   #
#  How Machine Learning Is Solving the Binary Function Similarity Problem.   #
#                                                                            #
#  MIT License                                                               #
#                                                                            #
#  Copyright (c) 2019-2022 Cisco Talos                                       #
#                                                                            #
#  Permission is hereby granted, free of charge, to any person obtaining     #
#  a copy of this software and associated documentation files (the           #
#  "Software"), to deal in the Software without restriction, including       #
#  without limitation the rights to use, copy, modify, merge, publish,       #
#  distribute, sublicense, and/or sell copies of the Software, and to        #
#  permit persons to whom the Software is furnished to do so, subject to     #
#  the following conditions:                                                 #
#                                                                            #
#  The above copyright notice and this permission notice shall be            #
#  included in all copies or substantial portions of the Software.           #
#                                                                            #
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,           #
#  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF        #
#  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND                     #
#  NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE    #
#  LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION    #
#  OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION     #
#  WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.           #
#                                                                            #
#  cli_flowchart.py - Call IDA_flowchart.py IDA script.                      #
#                                                                            #
##############################################################################

import click
import subprocess

from os import getcwd
from os import getenv
from os import walk
from os.path import abspath
from os.path import dirname
from os.path import isfile
from os.path import join
from os.path import relpath

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from multiprocessing import current_process

IDA_PATH = getenv("IDA_PATH", "/home/user/idapro-7.3/idat64")
IDA_PLUGIN = join(dirname(abspath(__file__)), 'IDA_flowchart.py')
REPO_PATH = dirname(dirname(dirname(abspath(__file__))))
LOG_PATH = "flowchart_log.txt"

def inner_walk(f_name, root, output_csv):

    output_csv = output_csv.replace('.csv', '_' + current_process().name+'.txt')
    
    if (not f_name.endswith(".i64")) and (not f_name.endswith(".idb")):
        return None

    idb_path = join(root, f_name)
    print("\n[D] Processing: {}".format(idb_path))

    if not isfile(idb_path):
        print("[!] Error: {} not exists".format(idb_path))
        return

    # Compute the normalized relative path from the main directory
    rel_idb_path = relpath(
        join(getcwd(), root, f_name),  # absolute path of IDB
        REPO_PATH)  # absolute path of the repo folder

    cmd = [IDA_PATH,
            '-A',
            '-L{}'.format(LOG_PATH),
            '-S{}'.format(IDA_PLUGIN),
            '-Oflowchart:{}:{}'.format(
                rel_idb_path,
                output_csv),
            idb_path]

    print("[D] cmd: {}".format(cmd))

    proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()

    return proc.returncode

@click.command()
@click.option("-i", "--idbs-folder", required=True,
              help="Path to the IDBs folder")
@click.option("-o", "--output-csv", required=True,
              help="Path to the output CSV file")
def main(idbs_folder, output_csv):
    """Call IDA_flowchart.py IDA script."""
    try:
        if not isfile(IDA_PATH):
            print("[!] Error: IDA_PATH:{} not valid".format(IDA_PATH))
            print("Use 'export IDA_PATH=/full/path/to/idat64'")
            return

        print("[D] IDBs folder: {}".format(idbs_folder))
        print("[D] Output CSV: {}".format(output_csv))

        export_success, export_error = 0, 0

        for root, _, files in walk(idbs_folder):
            with ProcessPoolExecutor(cpu_count()) as executor:
                futures = ( executor.submit(inner_walk, f_name, root, output_csv) for f_name in files )
                
                for future in as_completed(futures):
                    error = future.exception()
                    if error:
                        export_error += 1
                        print("Thread error while processing: ", future)
                        continue
                        
                    returncode = future.result()
                    if returncode is None:
                        pass
                    elif returncode == 0:
                        export_success += 1
                    else:
                        print("[!] Error in {} (returncode={})".format(
                            future, returncode))
                        export_error += 1

        print("\n# IDBs correctly processed: {}".format(export_success))
        print("# IDBs error: {}".format(export_error))

    except Exception as e:
        print("[!] Exception in cli_flowchart\n{}".format(e))


if __name__ == '__main__':
    main()
