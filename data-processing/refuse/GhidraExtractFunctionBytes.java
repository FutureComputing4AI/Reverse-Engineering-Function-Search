import java.io.File;

import ghidra.app.util.exporter.BinaryExporter;
import ghidra.app.script.GhidraScript;
import ghidra.program.model.address.*;
import ghidra.program.model.block.*;
import ghidra.program.model.lang.*;
import ghidra.program.model.listing.*;


public class GhidraExtractFunctionBytes extends GhidraScript {
	
    private static final String OUTPUT_DIR = "/path/to/output/directory";

    @Override
    public void run() throws Exception {

        FunctionManager functionManager = currentProgram.getFunctionManager();
        int functionCount = functionManager.getFunctionCount();
        System.out.println("Found " + functionCount + " functions");
        
		BinaryExporter binaryExporter = new BinaryExporter();
		int extractedFunctionsCount = 0;
		int failedFunctionsCount = 0;
		
		Address imageBase = currentProgram.getImageBase();
		
        for (Function function : functionManager.getFunctions(true)) {
            AddressSetView functionASV = function.getBody();
            
            long startRVA = functionASV.getMinAddress().subtract(imageBase);
            long endRVA = functionASV.getMaxAddress().subtract(imageBase);
            
            File bytesFile = new File(GhidraExtractFunctionBytes.OUTPUT_DIR + "/" + currentProgram.getExecutableSHA256() + 
                                     "\\" + function.getName() + "\\" + startRVA + "\\" + endRVA + "\\bytes.bin");
            boolean success = binaryExporter.export(bytesFile, currentProgram, functionASV, monitor);
            
            if (!success) {
            	System.out.println("Failed to export the function " + function.getName() + " in file " + currentProgram.getExecutableSHA256());
            	failedFunctionsCount++;
            }
            else {
            	extractedFunctionsCount++;
            }
            
        } // and all functions
	
	System.out.println("Extracted " + extractedFunctionsCount + " functions!");
	System.out.println("Failed on " + failedFunctionsCount + " functions.");
    }
}

