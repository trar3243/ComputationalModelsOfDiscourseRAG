import sys, os
import torch


CMDDROOT = os.environ['CMDDROOT'] # Expected to be a global environment variable. If not set, navigate to root of repository, call "source ./.env"
sys.path.append(CMDDROOT)

from ClassDefinition.Utils import Logger, ArgumentParser # type: ignore
from Main.segmenter import segment

required_arguments = []
optional_arguments = {
    "dataPath": f"{CMDDROOT}/Data/"
}
g_Logger = Logger(__name__)
g_ArgParse = ArgumentParser()
print = g_Logger.print

USAGE = """
main.py
    Required:
        
    Optional:
        dataPath=<dataPath>
"""


def initialize(inputArguments):
    print(f"ScriptName: {__file__}")
    try:
        g_ArgParse.setArguments(inputArguments, required_arguments, optional_arguments)
    except Exception as e:
        e.add_note(USAGE) # add usage note 
        raise
    g_ArgParse.set("device", "cuda" if torch.cuda.is_available() else "cpu")
    g_ArgParse.printArguments()



def main(inputArguments):
    initialize(inputArguments)
    
    segment(data_path=f"{g_ArgParse.get('dataPath')}/wikisection_en_disease_test.json", device=g_ArgParse.get("device"))



if __name__ == "__main__":
    try:
        main(sys.argv[1:]) # don't pass the script name to the argument parser 
    except Exception as e:
        g_Logger.logger.exception(e)
        exit(1)
