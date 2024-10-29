import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashSet;
import java.util.List;

import generic.lsh.vector.LSHVectorFactory;
import ghidra.app.script.GhidraScript;
import ghidra.features.bsim.gui.filters.BSimFilterType;
import ghidra.features.bsim.gui.filters.NotMd5BSimFilterType;
import ghidra.features.bsim.gui.search.results.BSimMatchResult;
import ghidra.features.bsim.query.FunctionDatabase;
import ghidra.features.bsim.query.FunctionDatabase.ErrorCategory;
import ghidra.features.bsim.query.facade.QueryDatabaseException;
import ghidra.features.bsim.query.facade.SFQueryInfo;
import ghidra.features.bsim.query.facade.SFQueryResult;
import ghidra.features.bsim.query.facade.SimilarFunctionQueryService;
import ghidra.features.bsim.query.protocol.BSimFilter;
import ghidra.program.database.symbol.FunctionSymbol;
import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.FunctionIterator;
import ghidra.util.exception.CancelledException;

public class MOTIFBSimQuery extends GhidraScript {

    // Threshold Settings
    private static final int MAX_NUM_FUNCTIONS = 31;
	private static final double SIMILARITY_BOUND = 0.0; 
	private static final double SIGNIFICANCE_BOUND = 0.0;

    private HashSet<FunctionSymbol> functionsToQuery;
    private SimilarFunctionQueryService queryService;
    private SFQueryInfo queryInfo;
    private BSimFilter bsimFilter;

    private String outputDir;
    private Path outputDirPath;
    private String databaseURL;

    @Override
    protected void run() throws Exception {

        /* When running on it's own in analyze headless mode use
         * $ ./analyzeHeadless <project location> <project name> \
            -import [file] \
            -postScript MOTIFBsimQuery <base output dir> <BSim URL> \
            -scriptPath <directory to this file>

         * To use with the mulithreader use something like
            mt_script = PostScriptMultithreader(
                binaries="/path/to/binaries/",
                ghidra_support_dir="/path/to/Ghidra/support/dir",
                ghidra_script_name="MOTIFBsimQuery",
                ghidra_script_path="/path/to/this/directory",
                ghidra_script_args=["/path/to/output/dir/", "<bsim URL>/<dbname>"],
                # other args as necessary...
            )
         */ 
        
        // Set up Command-line args
        String[] args = getScriptArgs();
        try {
            this.outputDir = args[0];
        } catch (Exception e) {
            throw new IllegalArgumentException("Missing output directory script parameter for SingleFileBSimQuery");
        }

        try {
            this.databaseURL = args[1];
        } catch (Exception e) {
            throw new IllegalArgumentException("Missing Bsim database URL in script parameter for SingleFileBSimQuery");
        }

        // Configure Inputs
        queryService = new SimilarFunctionQueryService(currentProgram);
        HashSet<FunctionSymbol> functionsToQuery = new HashSet<>();
        FunctionIterator functionIterator = currentProgram.getFunctionManager().getFunctionsNoStubs(true);
        for (Function func : functionIterator) {
            functionsToQuery.add((FunctionSymbol) func.getSymbol());
        }

        // Configure outputs
        String programSha = currentProgram.getExecutableSHA256();
        createLargeDatasetModeOutputDirPath(programSha);
        Files.createDirectories(this.outputDirPath);
        String outputFilePath = Paths.get(this.outputDirPath.toString(), programSha).toString() + ".txt";
        System.out.println("Saving to " + outputFilePath);
        File resultsFile = new File(outputFilePath);

        // Set up query
        queryInfo = new SFQueryInfo(functionsToQuery);
        bsimFilter = queryInfo.getBsimFilter();

        // Add threshold filters.
		queryInfo.setMaximumResults(MAX_NUM_FUNCTIONS);
		queryInfo.setSimilarityThreshold(SIMILARITY_BOUND);
		queryInfo.setSignificanceThreshold(SIGNIFICANCE_BOUND);

        // Connect to the database
        try {
            queryService.initializeDatabase(this.databaseURL);
			FunctionDatabase.Error error = queryService.getLastError();
			if (error != null && error.category == ErrorCategory.Nodatabase) {
				println("Database [" + this.databaseURL + "] cannot be found (does it exist?)");
				return;
			}
		}
		catch (QueryDatabaseException e) {
			println(e.getMessage());
			return;
		}

        // Execute the query and save
        List<BSimMatchResult> resultRows = executeQuery(queryInfo);

        // Example optional result filters that can be applied.
        // These filters will only be applied to the result set returned from the previous query.
        // exclude matching on other functions from this binary...
		addBsimFilter(new NotMd5BSimFilterType(), currentProgram.getExecutableMD5());

        List<BSimMatchResult> filteredRows =
        BSimMatchResult.filterMatchRows(bsimFilter, resultRows);
        printFunctionQueryResults(filteredRows, resultsFile);
    }
    

    private void createLargeDatasetModeOutputDirPath(String programSHA) {

        // if the SHA is 00112233445566778899aabb.. etc
        // the output dir becomes output_base_dir/00/11/22/33/44/55/66/77/88/99/aa/bb/...
        // to guarantee there are never more than 256 items in a folder

        String workingPath = this.outputDir;
        int index = 0;
        while (index < programSHA.length()) {
            String subset = programSHA.substring(index, index + 2);
            workingPath = Paths.get(workingPath, subset).toString();
            index += 2;
        }
        this.outputDirPath = Paths.get(workingPath);

    }


    /**
	 * Queries the database and returns the results. 
	 * 
	 * @param qInfo contains all information required for the query
	 * @return list of matches
	 * @throws QueryDatabaseException if there is a problem executing the query similar functions query
	 * @throws CancelledException if the user cancelled the operation
	 */
	private List<BSimMatchResult> executeQuery(SFQueryInfo qInfo)
        throws QueryDatabaseException, CancelledException {

        SFQueryResult queryResults = queryService.querySimilarFunctions(qInfo, null, monitor);
        List<BSimMatchResult> resultRows =
            BSimMatchResult.generate(queryResults.getSimilarityResults(), currentProgram);

        return resultRows;
    }

    /**
	 * Prints information about each function in the result set.
	 * 
	 * @param resultRows the list of rows containing the info to print
	 * @param title the title to print
	 */
	private void printFunctionQueryResults(List<BSimMatchResult> resultRows, File outFile) throws FileNotFoundException{
		
        try (PrintStream out = new PrintStream(new FileOutputStream(outFile))) {
            for (BSimMatchResult resultRow : resultRows) {
                String query = resultRow.getOriginalFunctionDescription().getExecutableRecord().getMd5();
                String match = resultRow.getMatchFunctionDescription().getExecutableRecord().getMd5();
                double sim = resultRow.getSimilarity();
                
                String result = query + "\\" + match + "\\" + String.valueOf(sim)  + "\n";
                // try (PrintStream out = new PrintStream(new FileOutputStream(outFile))) {
                //     out.print(result);
                //     out.flush();
                // }

                out.print(result);
                
            }

            out.close();
        }
	}

    /**
	 * Adds a filter to the given filter container.
	 * 
	 * @param filterTemplate the filter type to add
	 * @param value the value of the filter
	 */
	private void addBsimFilter(BSimFilterType filterTemplate, String value) {
		String[] inputs = value.split(",");
		for (String input : inputs) {
			if (!input.trim().isEmpty()) {
				bsimFilter.addAtom(filterTemplate, input.trim());
			}
		}
	}


}
