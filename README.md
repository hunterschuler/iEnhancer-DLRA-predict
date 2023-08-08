# iEnhancer-DLRA-predict
This is a prediction tool based on the iEnhancer-DLRA project by L. Zeng, et al. (2022) - https://github.com/lftxd1/iEnhancer-DLRA

This Python script (predict.py) will accept a FASTA input file and will make identification and classification predictions for potential DNA enhancers in those sequences.

The models included in this repository were trained with a modified version* of the scripts found in the original iEnhancer-DLRA repository. 

*The only changes I made to the original iEnhancer-DLRA code was exporting the models after they were trained, so those modified training scripts are not included in this repository.

# Environment
An environment.yml file is provided for you to recreate the environment I used to develop and run this script. Not all of the packages in the environment are strictly necessary though (e.g. jupyter-lab); the file is simply provided for convenience.

This script was developed using WSL2 (Ubuntu 22.04), partly becuase TensorFlow stopped supporting GPU functionality in native-Windows after version 2.10. You can probably run this script on Windows directly using TensorFlow's CPU functionality, or by downgrading TensorFlow to 2.10, but neither of these possibilities have been tested. 

# Arguments and Example Use
The script accepts the following arguments:

| Flag           | Required/Optional| Description                                                                                                           |
|----------------|------------------|------------------------------------------------------------------------------------------------------------------------|
| --input        |   [REQUIRED]     | Path to the input file. File must be in FASTA format.                                                                  |
| --model        |   [REQUIRED]     | Select whether you want the script to execute the classifier model ("c") or the identifier model ("i")                 |
| --step         |   [OPTIONAL]     | Dictates the step size for windowing input sequences longer than 200 bases. Acceptable range: 1-200.**                 |
| --output       |   [OPTIONAL]     | Set the name and destination for the output file. Default is "output/output_\<date-time\>.txt"                                  |
| --class_model  |   [OPTIONAL]     | Path to the classifier model (TensorFlow). This will default to the classifier model that is colocated with the script in "/classifier_model_trained" but optionally, you can input a path to a different TensorFlow model.|
| --ident_model  |   [OPTIONAL]     | Path to the identifier model (TensorFlow). This will default to the identifier model that is colocated with the script in "/identifier_model_trained" but optionally, you can input a path to a different TensorFlow model.|

** If the sequence length is not a multiple of the window size (i.e. if sequence_length % step != 0), then one additional window will be included that ends on the last base in the sequence.

Examples:
>python predict.py --input /your/path/inputfile.fa --model i
>
>python predict.py --input /your/path/inputfile.fa --model c --step 120 --output /your/path/sequence_xyz_output.txt

# Output
Expected output is a text file with one sequence per line. On each line, you should find the sequence name (assuming the sequences in your FASTA file were named or numbered), followed by an array with a prediction for each window of that sequence.

Example (one line):
>Hs2496; [0.0032715858, 0.94306844, 0.94309676, 0.94166934, 0.00023568304]
