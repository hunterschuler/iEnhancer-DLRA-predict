# 
#   Hunter Schuler
#   hschuler@smu.edu
#   Southern Methodist University
#   UT Southwestern Medical Center - Lin Xu Lab
#   06JULY2023
#
#   Based on code from the iEnhancer-DLRA tool:
#       L. Zeng, et al., 2022 (https://github.com/lftxd1/iEnhancer-DLRA)
#

import argparse
from datetime import datetime, timezone

import numpy as np
import tensorflow as tf
from Bio import SeqIO

from utils.MyUtils import *

# Get the current date and time in UTC
now = datetime.now()
UTC_now = now.astimezone(timezone.utc)
UTC_now = UTC_now.strftime('%Y-%m-%dT%H%M%SZ')

# Define the function to validate the step size
def validate_positive_integer(value):
    try:
        int_value = int(value)
        if int_value <= 0 or int_value > 200:
            raise argparse.ArgumentTypeError(f"\nThe value must a positive integer less than or equal to 200, but '{value}' was provided.\n")
        return int_value
    except ValueError:
        raise argparse.ArgumentTypeError(f"\nThe value must be a positive integer less than or equal to 200, but '{value}' was provided.\n")

# Define the function to validate the input file
def is_fasta(file_path):
    try:
        with open(file_path) as handle:
            try:
                records = SeqIO.parse(handle, "fasta")
                first_record = next(records)
                if bool(first_record.seq)==True:
                    return file_path
            except (ValueError, StopIteration):
                raise argparse.ArgumentTypeError(f"\nInput file error! \nVerify that the input is a FASTA file and that the file path is correct.\n(Also, don't use quotation marks for the filepath.)\n")
    except FileNotFoundError:
        raise argparse.ArgumentTypeError(f"\nInput file error! \nVerify that the input is a FASTA file and that the file path is correct.\n(Also, don't use quotation marks for the filepath.)\n")

# Define longer descriptions for the arguments
input_help = "[REQUIRED] Path to the input file. File must be in .FASTA format."
model_help = "[REQUIRED] Select whether you want the script to execute the classifier model (\"c\") or the identifier model (\"i\")"
step_help = "[OPTIONAL] Dictates the step size for windowing input sequences longer than 200 bases. E.g., 200 would have 0% overlap, 100 would have 50% overlap, 1 would have 99.5% overlap. Acceptable range: 1-200."
output_help = "[OPTIONAL] Set the name and destination for the output file. Default is \"/output_<date-time>.txt\""
class_model_help = "[OPTIONAL] Path to the classifier model (TensorFlow). This will default to the classifier model that is colocated with the script in \"/classifier_model_trained\" but optionally, you can input a path to a different TensorFlow model."
ident_model_help = "[OPTIONAL] Path to the identifier model (TensorFlow). This will default to the identifier model that is colocated with the script in \"/identifier_model_trained\" but optionally, you can input a path to a different TensorFlow model."

# Create input arguments
parser = argparse.ArgumentParser(description='Numeric argument validation script')

parser.add_argument('--input', type=is_fasta, help=input_help)
parser.add_argument('--model', help=model_help)
parser.add_argument('--step', type=validate_positive_integer, default=200, help=step_help)
parser.add_argument('--output', default=str('output_' + UTC_now + '.txt'), help=output_help)
parser.add_argument('--class_model', default='classifier_model_trained', help=class_model_help)
parser.add_argument('--ident_model', default='identifier_model_trained', help=ident_model_help)

# Parse the command line arguments
args = parser.parse_args()

# Access the arguments
input_step = args.step
input_file = args.input
input_classifier = args.class_model
input_identifier = args.ident_model
input_model = args.model
output_destination = args.output

# Define the function to load the saved model
def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

# Load the model depending on user argument
if input_model == 'c':
    model = load_model(input_classifier)
elif input_model == 'i':
    model = load_model(input_identifier)
else:
    raise Exception('\nInvalid model selection! Select either the classification model (\"--model c\") or the identification model (\"--model i\").\n')

# Define the dictionary for converting the sequence to numbers
dic={'A':'A','T':'T','G':'G','C':'C'}
dic_con={'A':1,'T':2,'G':3,'C':4}

# Define the function to generate the required input format for the model
def GenerateFromTextToNumpy(train):
    
    train_con = []
    train_text = []
    train_text_5 = []

    for i in train:
        t = threeSequence(i, 4)
        train_text.append(np.array(t))

        t_5 = threeSequence(i, 7) 
        train_text_5.append(np.array(t_5))

        con_t = [dic_con[key] for key in i]
        train_con.append(np.array(con_t))

    train_con = np.array(train_con)
    train_text = np.array(train_text)
    train_text_5 = np.array(train_text_5)

    print("Shape of train_con:", train_con.shape)

    return train_con, train_text, train_text_5

# Define the function to make predictions on new data
def make_predictions(model, new_data):
    
    # Preprocess the new data
    new_con, new_text, new_text_5 = GenerateFromTextToNumpy(new_data)

    # Make predictions
    predictions = model.predict({"con": new_con, "text": new_text, "text_5": new_text_5})
    predictions = tf.nn.sigmoid(predictions)
    predictions = predictions.numpy()
    predictions = predictions[:,0]
    return predictions

# Define the function to slice the sequence into windows
def slice_sequence(handed_sequence):
    full_steps = (len(handed_sequence.seq) // input_step) - (200/input_step) + 1
    if full_steps.is_integer() == False:
        raise Exception('\nSomething went wrong with the step size! The value for full_steps should have been an integer. This may be a logic error??\n')
    full_steps = int(full_steps)
    sequence_slices = []
    for i in range(full_steps):
        begin = i * input_step
        end = begin + input_step
        sequence_slices.append(str(handed_sequence.seq[begin:end]))
    if len(handed_sequence.seq) % input_step != 0:
        sequence_slices.append(str(handed_sequence.seq[(end-input_step):end]))
    return(sequence_slices)

# Define the main function
if __name__ == "__main__": 
    results = {}
    sequences = []
    for record in SeqIO.parse(input_file, "fasta"):
        # Extract and store each sequence from the FASTA file
        sequences.append(record)
    
    for i in range(len(sequences)):    
        prepped_sequence = slice_sequence(sequences[i])
        seq_id = sequences[i].id
        if seq_id in results:
            raise Exception('Error! There\'s already an identical sequence ID in the results. Check your data for duplicate sequences!')
        else:
            results[seq_id] = make_predictions(model, prepped_sequence)

# Write the results to a file
with open(output_destination, 'w') as f:
    for i in range(len(results)):
        f.write(list(results.keys())[i])
        f.write('; ')
        f.write(str(list(list(results.values())[i])))
        f.write("\n")
