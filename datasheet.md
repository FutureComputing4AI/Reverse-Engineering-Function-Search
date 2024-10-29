# Datasheet for REFuSe-Bench

## Motivation

### For what purpose was the dataset created? 

Though significant research effort has been spent creating machine learning tools for binary function similarity detection (BFSD), existing BFSD datasets do not mimic the real-world scenarios encountered by security practioners. In particular, most prior datasets consist of small numbers of benign Linux binaries from standard packages like *openssl* and *coreutils*, which are not representative of the available landscape of executables. Thus, it is difficult to accurately assess the success of machine learning models on BFSD. REFuSe-Bench addresses this challenge by assembling a suite of five datasets that together reflect the true diversity of BFSD applications. The benchmark covers datasets of both Windows and Linux binaries, including some historical datasets for comparison purposes, and also captures standard libraries, user code from GitHub, and real malware. With nearly 150,000 binaries, REFuSe-Bench is the largest and most comprehensive BFSD benchmark published to date.

### Who created the dataset (e.g., which team, research group) and on behalf of which entity (e.g., company, institution, organization)?

REFuSe-Bench was created by Rebecca Saul and Edward Raff of Booz Allen Hamilton on behalf of the Laboratory for Physical Sciences. 

### Who funded the creation of the dataset? 

REFuSe-Bench was funded by the Laboratory for Physical Sciences.

## Composition

### What do the instances that comprise the dataset represent (e.g., documents, photos, people, countries)?

REFuSe-Bench consists of five datasets of binary executables. 

### How many instances are there in total (of each type, if appropriate)?

The datasets in REFuSe-Bench break down as follows:

| Dataset | OS | No.Binaries | Source |  
| ------- | --- | ----------- | ------- |  
| Assemblage | Windows | 135,975 | User code from GitHub |  
| MOTIF | Windows | 3,095 | Malware |  
| Common Libraries | Windows | 40 | Select standard libraries |   
| Marcelli Dataset-1 | Linux | 919 | Select standard libraries |  
| BinaryCorp | Linux | 9,675 | ArchLinux, Arch User Repository |   

### Does the dataset contain all possible instances or is it a sample (not necessarily random) of instances from a larger set?

REFuSe-Bench does not incorporate every dataset that has been used in the context of BFSD, as many of these datasets have significant overlap in both their coverage and their blind spots. To address these weaknesses, and provide a more diverse and more representative set of tasks for BFSD, REFuSe-Bench assimilates two prominent Linux datasets from the BFSD literature, while also integrating three datasets of Windows binaries that have not previously been used for BFSD. 

### What data does each instance consist of? 

Each dataset in REFuSe-Bench is a dataset of binary executables.

### Is there a label or target associated with each instance?

N/A

### Is any information missing from individual instances?

No

### Are relationships between individual instances made explicit (e.g., users’ movie ratings, social network links)?

N/A

### Are there recommended data splits (e.g., training, development/validation, testing)?

As part of REFuSe-Bench, we also curated a training dataset of Assemblage data. The Assemblage training and testing splits contain completely separate GitHub projects. Additionally, we deduplicated binary functions according to a hash of their bytes, and further ensured that common functions (functions that appear in more than half of all binaries) are not present in both the training and the testing split. More information about our training and testing split can be found in Section 3.1 of our paper.

REFuSe-Bench includes the test datasets from BinaryCorp and Marcelli Dataset-1. These datasets are also accompanied by training data splits, though we do not directly leverage them in this work.

### Are there any errors, sources of noise, or redundancies in the dataset?

To the best of our knowledge, there is no overlap between the datasets in REFuSe-Bench.

### Is the dataset self-contained, or does it link to or otherwise rely on external resources (e.g., websites, tweets, other datasets)?

N/A

### Does the dataset contain data that might be considered confidential (e.g., data that is protected by legal privilege or by doctor-patient confidentiality, data that includes the content of individuals’ non-public communications)?

No, the benign binaries in the datasets are compiled from publicly available source code, and malware is not protected by any license. 

### Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety?

No

### Does the dataset relate to people? 

No

### Does the dataset identify any subpopulations (e.g., by age, gender)?

N/A

### Is it possible to identify individuals (i.e., one or more natural persons), either directly or indirectly (i.e., in combination with other data) from the dataset?

N/A

### Does the dataset contain data that might be considered sensitive in any way (e.g., data that reveals racial or ethnic origins, sexual orientations, religious beliefs, political opinions or union memberships, or locations; financial or health data; biometric or genetic data; forms of government identification, such as social security numbers; criminal history)?

N/A

## Collection process

### How was the data associated with each instance acquired?

In the benign datasets, binaries were compiled from source, and metadata about the compilation was recorded at that time. For the MOTIF dataset, binaries, and their associated metadata, were collected by surveying open-source threat intelligence reports.

### What mechanisms or procedures were used to collect the data (e.g., hardware apparatus or sensor, manual human curation, software program, software API)?

The Assemblage and Common Libraries datasets were built using the [Assemblage tool](https://arxiv.org/pdf/2405.03991), a distributed system for collecting and building Windows PE binaries. The MOTIF dataset was collected based on binary hashes included in open-source threat intelligence reports. The Marcelli Dataset-1 and BinaryCorp datasets were constructed using procedures published in prior works, see [Marcelli](https://www.usenix.org/system/files/sec22-marcelli.pdf) and [Wang](https://arxiv.org/pdf/2205.12713).

### If the dataset is a sample from a larger set, what was the sampling strategy (e.g., deterministic, probabilistic with specific sampling probabilities)?

The datasets included in REFuSe-Bench were manually selected to be representative of a variety of real-world binary function similarity detection applications. 

### Who was involved in the data collection process (e.g., students, crowdworkers, contractors) and how were they compensated (e.g., how much were crowdworkers paid)?

N/A

### Over what timeframe was the data collected?

The MOTIF binaries were collected between 2016 and 2021. Marcelli Dataset-1 and BinaryCorp were both released in 2022. Binaries in the Assemblage dataset were built between 2022 and 2023. Finally, the Common Libraries dataset was built in 2024.

### Were any ethical review processes conducted (e.g., by an institutional review board)?

No

### Does the dataset relate to people?

No

### Did you collect the data from the individuals in question directly, or obtain it via third parties or other sources (e.g., websites)?

N/A

### Were the individuals in question notified about the data collection?

N/A

### Did the individuals in question consent to the collection and use of their data?

N/A

### If consent was obtained, were the consenting individuals provided with a mechanism to revoke their consent in the future or for certain uses?

N/A

### Has an analysis of the potential impact of the dataset and its use on data subjects (e.g., a data protection impact analysis) been conducted?

N/A

### Any other comments?

## Preprocessing/cleaning/labeling

### Was any preprocessing/cleaning/labeling of the data done (e.g., discretization or bucketing, tokenization, part-of-speech tagging, SIFT feature extraction, removal of instances, processing of missing values)?

No

### Was the “raw” data saved in addition to the preprocessed/cleaned/labeled data (e.g., to support unanticipated future uses)?

N/A

### Is the software used to preprocess/clean/label the instances available?

N/A

### Any other comments?

## Uses

### Has the dataset been used for any tasks already?

REFuSe-Bench has been used to evaluate three machine learning models for binary function similarity detection. 

### Is there a repository that links to any or all papers or systems that use the dataset?

No. The code accompanying the REFuSe-Bench paper is not yet publicly available, but there are plans to release it in the near future. 

### What (other) tasks could the dataset be used for?

The datasets in REFuSe-Bench may be useful for a variety of other binary analysis tasks, including, but not limited to, function boundary identification and compiler provenance identification.

### Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses?

No

### Are there tasks for which the dataset should not be used?

No

## Distribution

### Will the dataset be distributed to third parties outside of the entity (e.g., company, institution, organization) on behalf of which the dataset was created? 

Yes. Three of the datasets that make up REFuSe-Bench are already publicly available in their entirety, and for the other two, we provide detailed instructions on how to reproduce the binaries from the source code.

### How will the dataset will be distributed (e.g., tarball on website, API, GitHub)?

Instructions for testing new models on REFuSe-Bench will be hosted on GitHub.

### When will the dataset be distributed?

REFuSe-Bench should be publicly released by December 2024.

### Will the dataset be distributed under a copyright or other intellectual property (IP) license, and/or under applicable terms of use (ToU)?

The code for conducting experiments with REFuSe-Bench will be published under the MIT license. The binaries that make up the REFuSe-Bench datasets are each subject to their own original licenses.

### Have any third parties imposed IP-based or other restrictions on the data associated with the instances?

Our benchmark relies on a number of prior released datasets/binaries, which we are careful to access and recommend accessing in a manner that abides by and respects their licenses. For our developed code and benchmark, there is no encumberment of third-party IP or other restrictions, and it is made available under an OSS license. 

### Do any export controls or other regulatory restrictions apply to the dataset or to individual instances?

Standard US law applies to the products of this work performed in the US with U.S.G. federal funding. 

## Maintenance

### Who is supporting/hosting/maintaining the dataset?

The benchmark will be hosted on Github. The prior datasets used to create the benchmark are under their own respective maintenance/auspices. 

### How can the owner/curator/manager of the dataset be contacted (e.g., email address)?

Via email addresses and the Github repo that will be made public upon publication of the manuscript. 

### Is there an erratum?

Not at this time.

### Will the dataset be updated (e.g., to correct labeling errors, add new instances, delete instances)?

Dataset updates will be dependent upon community feedback. Any updates will be kept strictly separate to avoid confusion where the ``same'' datasets at different points in time would produce different results. While we are open to updates, we are also aware from prior experience that making a dataset too large can inhibit researchers with limited computational throughput. So, we will wait to see how the target community handles the size of the current dataset before committing to larger updates. 

Any legally required alterations in the original jurisdiction will be followed. 

### If the dataset relates to people, are there applicable limits on the retention of the data associated with the instances (e.g., were individuals in question told that their data would be retained for a fixed period of time and then deleted)?

N/A

### Will older versions of the dataset continue to be supported/hosted/maintained?

Our contribution is the benchmarks, and older benchmarks will be supported on a best-effort basis by the authors. The datasets used in the benchmark are subject to their own respective maintenance plans. 

### If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so?

We welcome others to contribute to REFuSe-Bench through pull requests on GitHub. 
