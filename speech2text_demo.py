from transformers import pipeline
import argparse

# Instantiate the parser
parser = argparse.ArgumentParser(description='Optional app description')

# Required positional argument
parser.add_argument('file_name', type=str,
                    help='Input the name of the file to trascribe')

# parse the arguments
args = parser.parse_args()

# speech2text pipeline
pipe = pipeline("automatic-speech-recognition", "facebook/wav2vec2-base-960h")
result = pipe(args.file_name)

print(result)