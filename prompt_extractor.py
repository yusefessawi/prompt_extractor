import tensorflow
import pandas as pd
import argparse

# tf.saved_model.save(model, 'path_to_saved_model')

# # Load the model
# loaded_model = tf.saved_model.load('path_to_saved_model')

def main():
    parser = argparse.ArgumentParser(description= "Specify a LLM-generated response or file and (optionally) the model that generated it")
    parser.add_argument('response', type = str, help="LLM-generated response")
    args = parser.parse_args()
    print(args.response)

main()

