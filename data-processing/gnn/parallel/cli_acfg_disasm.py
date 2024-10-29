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
#  cli_acfg_disasm.py - Call IDA_acfg_disasm.py IDA script.                  #
#                                                                            #
##############################################################################

import click
import json
import subprocess
import time

from os import getenv
from os.path import abspath
from os.path import dirname
from os.path import isfile
from os.path import join

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

IDA_PATH = getenv("IDA_PATH", "/home/user/idapro-7.3/idat64")
IDA_PLUGIN = join(dirname(abspath(__file__)), 'IDA_acfg_disasm.py')
REPO_PATH = dirname(dirname(dirname(abspath(__file__))))
LOG_PATH = "acfg_disasm_log.txt"

def inner_loop(idb_rel_path, json_path, output_dir):

    # Convert the relative path into a full path
    idb_path = join(REPO_PATH, idb_rel_path)

    if not isfile(idb_path):
        return 1

    cmd = [IDA_PATH,
            '-A',
            '-L{}'.format(LOG_PATH),
            '-S{}'.format(IDA_PLUGIN),
            '-Oacfg_disasm:{}:{}:{}'.format(
                json_path,
                idb_rel_path,
                output_dir),
            idb_path]

    proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()

    return proc.returncode

@click.command()
@click.option('-j', '--json-path', required=True,
              help='JSON file with selected functions.')
@click.option('-o', '--output-dir', required=True,
              help='Output directory.')
def main(json_path, output_dir):
    """Call IDA_acfg_disasm.py IDA script."""
    try:
        if not isfile(IDA_PATH):
            print("[!] Error: IDA_PATH:{} not valid".format(IDA_PATH))
            print("Use 'export IDA_PATH=/full/path/to/idat64'")
            return

        print("[D] JSON path: {}".format(json_path))
        print("[D] Output directory: {}".format(output_dir))

        if not isfile(json_path):
            print("[!] Error: {} does not exist".format(json_path))
            return

        with open(json_path) as f_in:
            jj = json.load(f_in)

            success_cnt, error_cnt = 0, 0
            start_time = time.time()
             
            with ProcessPoolExecutor(cpu_count()) as executor:
                futures = ( executor.submit(inner_loop, idb_rel_path, json_path, output_dir) for
                            idb_rel_path in jj.keys() )

                for future in as_completed(futures):

                    error = future.exception()
                    if error:
                        print("Thread error while processing: ", future)
                        continue

                    print(future.result())
                    if future.result()==0:
                        success_cnt += 1
                    else:
                        error_cnt += 1

            end_time = time.time()
            print("[D] Elapsed time: {}".format(end_time - start_time))
            with open(LOG_PATH, "a+") as f_out:
                f_out.write("elapsed_time: {}\n".format(end_time - start_time))

            print("\n# IDBs correctly processed: {}".format(success_cnt))
            print("# IDBs error: {}".format(error_cnt))

    except Exception as e:
        print("[!] Exception in cli_acfg_disasm\n{}".format(e))


if __name__ == '__main__':
    main()
