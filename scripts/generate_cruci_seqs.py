"""
DNA sequence generation using Evo model with FASTA or CSV output.

Usage: 
    From file: python generate_and_save.py --input_file prompts.csv --model_name model_name
    From string: python generate_and_save.py --prompt "ATCG..." --model_name model_name
"""
from pathlib import Path
from typing import List, Union, NamedTuple, Optional
import logging
import uuid
import csv
import os
import argparse
from Bio.Seq import Seq
import numpy as np

# from stripedhyena.model import StripedHyena
# from stripedhyena.tokenizer import CharLevelTokenizer
from eval.models import load_model

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelOutput(NamedTuple):
    sequences: List[str]
    scores: List[float]


def adjust_prompt_length(prompt: str, percent: Optional[float] = None, length: Optional[int] = None) -> str:
    """Adjust prompt length based on percent and length parameters."""
    if percent is not None or length is not None:
        if percent is not None:
            adjusted_length = int(len(prompt) * percent)
        else:
            adjusted_length = len(prompt)
        if length is not None:
            prompt = prompt[:min(adjusted_length, length)]
        else:
            prompt = prompt[:adjusted_length]
    return prompt


def read_prompts_from_file(input_file: Path, percent: Optional[float] = None, length: Optional[int] = None) -> List[str]:
    """Read prompts from CSV or FASTA/FNA file and adjust them based on percent and length if provided."""
    file_type = input_file.suffix[1:].lower()
    prompts = []
    
    if file_type == 'csv':
        with open(input_file, encoding='utf-8-sig', newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header
            for row in reader:
                prompt = row[0]
                prompts.append(adjust_prompt_length(prompt, percent, length))
    
    elif file_type in ['fasta', 'fa', 'fna']:
        current_sequence = []
        with open(input_file) as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_sequence:  # Save the previous sequence if it exists
                        prompt = ''.join(current_sequence)
                        prompts.append(adjust_prompt_length(prompt, percent, length))
                        current_sequence = []
                else:
                    current_sequence.append(line)
            
            # Don't forget the last sequence
            if current_sequence:
                prompt = ''.join(current_sequence)
                prompts.append(adjust_prompt_length(prompt, percent, length))
    
    return prompts

# def model_load(model_name: str) -> tuple[StripedHyena, CharLevelTokenizer]:
#     """Load the model and tokenizer."""
#     evo_model = load_model(model_name)
#     return evo_model.model, evo_model.tokenizer

def model_load(model_name: str) -> tuple:
    """Load the model and tokenizer."""
    evo_model = load_model(model_name)
    return evo_model.model, evo_model.tokenizer

def generate_sequences(
    prompts: Union[str, List[str]],
    model: any,
    name: str,
    tokenizer: any,
    n_tokens: int = 1000,
    temperature: float = 0.7,
    top_k: int = 4,
    device: str = 'cuda:0',
    n_sample_per_prompt: int = 1,
) -> ModelOutput:
    """Generate sequences from prompts."""
    if 'evo2' in name:
        from eval.vortex_generation import generate
    else:
        from eval.generation import generate
    # if isinstance(prompts, str): # put this logic in main 
    #     prompts = [prompts]
    sequences, scores = generate(
        prompt_seqs=prompts,
        model=model,
        tokenizer=tokenizer,
        n_tokens=n_tokens,
        temperature=temperature,
        top_k=top_k,
        batched=True,
        device=device,
        force_prompt_threshold=2,
        cached_generation=True,
        verbose=True
    ) # takes a list of strings and will return 
    return ModelOutput(sequences, scores)

def save_sequences_fasta(sequences: List[str], scores: List[float], output_file: str, 
                        prompt_file: Optional[str] = None, hyperparameters: Optional[dict] = None, short_header = False):
    """
    Save generated sequences to a FASTA file with scores, prompt file, and hyperparameters in the headers.
    
    Args:
        sequences: List of sequence strings
        scores: List of corresponding scores
        output_file: Path to output FASTA file
        prompt_file: Optional name of prompt file used
        hyperparameters: Optional dictionary of hyperparameters used
    """
    with open(output_file, 'a') as f:
        for idx, (seq, score) in enumerate(zip(sequences, scores)):
            # Create base FASTA header with sequence number and score
            if short_header:
                header = f">sequence_{idx+1}"
            else:
                header = f">sequence_{idx+1}_score_{score:.4f}"
            
            # Add prompt file info if provided
            if prompt_file:
                header += f" prompt_file={prompt_file}"
                
            # Add hyperparameters if provided
            if hyperparameters:
                header += f" hyperparameters={str(hyperparameters)}"
                
            # Write header
            f.write(f"{header}\n")
            f.write(f"{seq}\n")


def save_sequences_csv(prompts: List[str],sequences: List[str], scores: List[float], output_file: str, prompt_file: Optional[str] = None, hyperparameters: Optional[dict] = None):
    """Save generated sequences, their scores, prompt file name, and hyperparameters to a CSV file."""

    with open(output_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not os.path.exists(output_file):
            writer.writerow(['UUID','Prompt','Generated Sequence','Hyperparameters'])
        for prompt,seq, score in zip(prompts,sequences, scores):
            writer.writerow([uuid.uuid4().hex, prompt,seq, hyperparameters])

def save_sequences_csv_no_score(prompts: List[str],sequences: List[str],uuids: List[str],descriptions: List[str], output_file: str, prompt_file: Optional[str] = None, hyperparameters: Optional[dict] = None):
    """Save generated sequences, their scores, prompt file name, and hyperparameters to a CSV file."""

    with open(output_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if os.stat(output_file).st_size == 0:
            writer.writerow(['UUID','Prompt','Generated Sequence', 'Description', 'Hyperparameters'])
        for prompt,seq,uuid, desc in zip(prompts,sequences, uuids, descriptions):
            writer.writerow([uuid, prompt,seq, desc, hyperparameters])


def save_sequences(prompt:Union[List[str],str],sequences: List[str], scores: List[float], output_file: str,
                  prompt_file: Optional[str] = None, hyperparameters: Optional[dict] = None):
    """
    Save sequences in either FASTA or CSV format based on file extension.
    
    Args:
        sequences: List of sequence strings
        scores: List of corresponding scores
        output_file: Path to output file
        prompt_file: Optional name of prompt file used
        hyperparameters: Optional dictionary of hyperparameters used
    """
    file_ext = Path(output_file).suffix.lower()
    
    if file_ext == '.fasta' or file_ext == '.fa' or file_ext == '.fna':
        save_sequences_fasta(sequences, scores, output_file, prompt_file, hyperparameters)
        logger.info(f"Saved sequences in FASTA format to {output_file}")
    elif file_ext == '.csv':
        save_sequences_csv(prompt,sequences, scores, output_file, prompt_file, hyperparameters)
        logger.info(f"Saved sequences in CSV format to {output_file}")
    else:
        logger.warning(f"Unrecognize  '{file_ext}'. Defaulting to csv format.")
        save_sequences_csv(sequences, scores, output_file, prompt_file, hyperparameters)

def save_sequences_no_score(prompt:Union[List[str],str],sequences: List[str], uuids: List[str], descriptions: List[str], output_file: str,
                  prompt_file: Optional[str] = None, hyperparameters: Optional[dict] = None):

    # THIS IS JUST TEMPORARY/ made quickly for a single purpose. would do save_sequences if it owrks 
    """
    Save sequences in either FASTA or CSV format based on file extension.
    
    Args:
        sequences: List of sequence strings
        scores: List of corresponding scores
        output_file: Path to output file
        prompt_file: Optional name of prompt file used
        hyperparameters: Optional dictionary of hyperparameters used
    """
    file_ext = Path(output_file).suffix.lower()
    
    if file_ext == '.csv':
        save_sequences_csv_no_score(prompt,sequences, uuids, descriptions, output_file, prompt_file, hyperparameters)
        logger.info(f"Saved sequences in CSV format to {output_file}")


def read_prompts(input_file: str, batched: bool = True, batch_size: int = 4) -> np.array:
    if batched: 
        promptseqs = []
        prompt_descs = []
        uuids = []
        with open(input_file, encoding='utf-8-sig', newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                promptseqs.append(row[0])
                prompt_descs.append(row[1]) # this will be the description 
                uuids.append(uuid.uuid4().hex) # this will be the uuid
        # will want to map uuid to both ... and maybe prompt to uuid to check it 
        promptseqs = np.array(promptseqs)
        prompt_descs = np.array(prompt_descs)
        uuids = np.array(uuids)
        # if we batch prompts, we need a way to keep track of the original label

        # Initialize dictionary to hold sequences grouped by length
        prompt_split_str = {}
        for idx, string in enumerate(promptseqs):
            length = len(string)
            if length not in prompt_split_str:
                prompt_split_str[length] = []
            prompt_split_str[length].append((string, prompt_descs[idx], uuids[idx]))

            if len(prompt_split_str[length]) == batch_size:
                if 'batches' not in prompt_split_str:
                    prompt_split_str['batches'] = []
                prompt_split_str['batches'].append(prompt_split_str[length])
                prompt_split_str[length] = []  # Reset the list for new batches of this length
        for key, value in prompt_split_str.items():
            if len(value) > 0 and key != 'batches':  # Exclude the 'batches' key from being re-added
                if 'batches' not in prompt_split_str:
                    prompt_split_str['batches'] = []
                prompt_split_str['batches'].append(value)
        

        # for string in promptseqs:
        #     length = len(string)
        #     if length not in prompt_split_str:
        #         prompt_split_str[length] = []
        #     prompt_split_str[length].append(string)
        
        #     # Check if the current list has reached the maximum batch size
        #     if len(prompt_split_str[length]) == batch_size:
        #         if 'batches' not in prompt_split_str:
        #             prompt_split_str['batches'] = []
        #         prompt_split_str['batches'].append(prompt_split_str[length]) # batches hold a list of lists when they meet the full batch size 
        #         prompt_split_str[length] = []  # Reset the list for new batches of this length

        # # Check for any remaining sequences that haven't been added to batches
        # # these will be ones that aren't full 
        # for key, value in prompt_split_str.items():
        #     if len(value) > 0 and key != 'batches':  # Exclude the 'batches' key from being re-added
        #         if 'batches' not in prompt_split_str:
        #             prompt_split_str['batches'] = []
        #         prompt_split_str['batches'].append(value)

        # Return only the batches if they exist; this will return string, description, and uuid
        return prompt_split_str.get('batches', [])
    else: 
        promptseqs = []
        with open(input_file, encoding='utf-8-sig', newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                promptseqs.append(row[0], row[1], uuid.uuid4().hex) # this will be the description
        promptseqs = np.array(promptseqs)
        #seq_ids = np.array(seq_ids)
        #promptseqs = [' '.join(inner_list) for inner_list in promptseqs]
        return promptseqs # this will be a 
    # this funciton will either a list of lists (which each inside list is a batch of sequences), or a list of strings 

def main():
    parser = argparse.ArgumentParser(description='Generate DNA sequences from prompts')
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input_file', type=str, help='Input CSV file containing prompts')
    input_group.add_argument('--prompt', type=str, help='Single DNA prompt sequence')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model to use')
    parser.add_argument('--output', type=str, default='generated_sequences.fasta', 
                      help='Output file path (use .fasta/.fa for FASTA format or .csv for CSV format)')
    parser.add_argument('--n_tokens', type=int, default=1000, help='Number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=1.0, help='Generation temperature')
    parser.add_argument('--top_k', type=int, default=4, help='Top-k sampling parameter')
    parser.add_argument('--num_generations', type=int, default=3)
    parser.add_argument('--prompt_len', type=int, required=False, help='Length of prompt to use, otherwise uses whole sequence in prompt file')
    parser.add_argument('--prompt_percent', type=float, required=False, help='Percent of prompt to use, otherwise uses whole sequence in prompt file')
    parser.add_argument('--batched', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default=10)
    args = parser.parse_args()

    hyperparameters = {
        'model_name': args.model_name,
        'prompt_len': args.prompt_len,
        'prompt_percent': args.prompt_percent,
        'n_tokens': args.n_tokens,
        'temperature': args.temperature,
        'top_k': args.top_k,
        'batch_size': args.batch_size,
        'batched': args.batched
    }
    
    # Load model
    logger.info(f"Loading model {args.model_name}")
    model, tokenizer = model_load(args.model_name)


    
    # Get prompts
    if args.input_file: # when prompts are an input file
        # get these from arg
        prompts = read_prompts(Path(args.input_file), batched =args.batched, batch_size= args.batch_size) 
        # prompt_strings = prompts[:,:, 0] # this will be a list of strings
        # will return a list of lists of lists, where the most inner list is the [prompt, uid, description] or just a list of lists depending on batched or not 
        logger.info(f"Loaded {len(prompts)} prompts from file") # technically prompts would be nthe number of batches ?? 
    else: # when prompts are a single prompt
        prompts = [[args.prompt, "no_description", uuid.uuid4().hex]] # nest becuase it is easier 
        # prompt_strings = [args.prompt] # this will be a list of strings
        logger.info("Using single prompt")
    
    # so here, prompts can be a list of list of strings, a list of strings, or a sring 
    
    # Generate sequences
    logger.info("Generating sequences")
    for i in range(args.num_generations): # how many generations per string 
        if type(prompts[0][0]) == str: 
            prompts = [prompts] # we meed to make it 3d 
        for prompt in prompts: 
            prompt_strings = [row[0] for row in prompt]
            prompt_descs = [row[1] for row in prompt]
            prompt_uids = [row[2] for row in prompt]
        
            # this will either be a string or list of strings
            # if type(prompt) == str: # if the prompt is a string, we need to make it a list of strings
            #     prompt = [prompt] # wouldn't need to batch this if only one string given
            output = generate_sequences(
                prompts=prompt_strings,
                model=model,
                name=args.model_name,
                tokenizer=tokenizer,
                n_tokens=args.n_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )
            # output will be a list of sequences for each prompt in promt 
            assert(len(output.sequences) == len(prompt)), f"Output sequences {len(output.sequences)} do not match input prompts {len(prompt)}"
            results = []
            for idx in range(len(prompt)):
                single_generated_sequence = output.sequences[idx]
                # single_score = output.scores[idx]
                # now we have the prompt and the generated sequence paired 
                results.append([prompt_strings[idx], single_generated_sequence, prompt_uids[idx], prompt_descs[idx]])
            # saving sequs, geneated, uid, description, outupt, input, hypeerparam
      
            save_sequences_no_score([row[0] for row in results], [row[1] for row in results], [row[2] for row in results],[row[3] for row in results] , args.output, args.input_file, hyperparameters)


if __name__ == "__main__":
    main()

