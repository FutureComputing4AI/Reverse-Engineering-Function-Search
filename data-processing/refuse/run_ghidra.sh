#!/bin/bash

for ff in $(find /path/to/binary/dataset/ -type f);
do
java -jar ghidra.jar /tmp GhidraTempProject -import $ff -postScript GhidraExtractFunctionBytes -scriptPath . -analysisTimeoutPerFile 60 -deleteProject
done
