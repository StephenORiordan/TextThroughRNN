# Importing the textgenrnn module
from textgenrnn import textgenrnn
import time

# Importing the model
textgen = textgenrnn(weights_path='Complete Model/Complete Model_weights.hdf5',
                    vocab_path='Complete Model/Complete Model_vocab.json',
                    config_path='Complete Model/Complete Model_config.json')

# Wait a second for all the TensorFlow dependences to list
time.sleep(1)

# Ask if you want to use interactive mode
interactive = input("Do you want to use interactive mode? (True or False) ")

# Check if a valid input has been entered
while interactive.lower() != "true" and interactive.lower() != "false":
    print("Please select True or False")
    interactive = input("Do you want to use interactive mode? (True or False) ")

# Converts to Boolean rather than String
if interactive.lower() == "true":
    interactive = True
    
if interactive == "false":
    interactive = False

# Asks other inputs
temp = []
temp.append(float(input("What temperature do you want to generate sentences with? ")))
print(temp)
n = int(input ("How many sentences do you want to generate? "))

# Generating the sentences
if interactive == True:
    textgen.generate(interactive=True, top_n=n, temperature = temp)

else:
    textgen.generate_samples(n, temp)