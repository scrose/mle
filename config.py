'''
Configuration Settings
--------------------------
'''

import argparse


# ----------------------------------------
# Global variables within this script
arg_lists = []
parser = argparse.ArgumentParser()


# ----------------------------------------
# Some nice macros to be used for arparse
def str2bool(v):
    return v.lower() in ("true", "1")


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


# ----------------------------------------
# Arguments for the main program
main_arg = add_argument_group("Main")


main_arg.add_argument("--mode", type=str,
                      default="train",
                      choices=["train", "preprocess", "test"],
                      help="Run mode")

main_arg.add_argument('--h', type = str, 
                       default= '', 
                       help = 'Show configuration parameters.')
# ----------------------------------------
# Arguments for training
train_arg = add_argument_group("Training")

train_arg.add_argument('--pretrained', type = str, 
                       default = '',
                       help = 'Path to the pretrained word embedding')

train_arg.add_argument('--train_raw', type = str, 
                       default= './data/raw/train-v1.1.json', 
                       help = 'Path to the training raw data')
        
train_arg.add_argument('--valid_raw', type = str, 
                       default = './data/raw/dev-v1.1.json', 
                       help = 'Path to the validation raw data')

train_arg.add_argument('--test_raw', type = str, 
                       default = './data/raw/dev-v1.1.json', 
                       help = 'Path to the test raw data')

train_arg.add_argument('--train_data', type = str, 
                       default= './data/processed/train.csv', 
                       help = 'Path to the processed training data')
        
train_arg.add_argument('--valid_data', type = str, 
                       default = './data/processed/valid.csv', 
                       help = 'Path to the processed validation data.')

train_arg.add_argument('--test_data', type = str, 
                       default = './data/processed/test.csv', 
                       help = 'Path to the processed test data.')

train_arg.add_argument("--batch_size", type=int,
                       default=64,
                       help="Size of each training batch")

train_arg.add_argument("--num_epochs", type=int,
                       default=10,
                       help="Number of epochs to train")

train_arg.add_argument("--val_intv", type=int,
                       default=1000,
                       help="Validation interval")

train_arg.add_argument("--rep_intv", type=int,
                       default=1000,
                       help="Report interval")

train_arg.add_argument("--save_dir", type=str,
                       default="./save",
                       help="Directory to save best model and state params")

train_arg.add_argument("--log_dir", type=str,
                       default="./logs",
                       help="Directory to save logs and current model")

train_arg.add_argument("--eval_results", type=str,
                       default="./eval/results.json",
                       help="Directory to save evaluation results")

train_arg.add_argument("--resume", type=str2bool,
                       default=True,
                       help="Whether to resume training from existing checkpoint")
    
# ----------------------------------------
# Arguments for model
model_arg = add_argument_group("Model")



def get_config():
    config, unparsed = parser.parse_known_args()

    return config, unparsed


def print_usage():
    parser.print_usage()

#
# config.py ends here
