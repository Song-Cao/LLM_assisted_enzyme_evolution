import csv
import requests
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from difflib import SequenceMatcher
from Bio import pairwise2
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import itertools



THRESHOLD_TANIMOTO = 0.6
LEVENSHTEIN_THRESHOLD = 40
SEQUENCE_LENGTH_THRESHOLD = 600
TRAIN_TEST_RATIO = 0.8
MAX_OCCURRENCES = 5
SEQUENCE_IDENTITY_THRESHOLD = 0.6

def get_kegg_reaction_smiles(reaction_id):
    """Retrieve reactant and product SMILES for a given KEGG reaction ID."""
    url = f"http://rest.kegg.jp/get/{reaction_id}"
    response = requests.get(url)
    if response.status_code == 200:
        lines = response.text.splitlines()
        reactants, products = None, None
        for line in lines:
            if line.startswith("EQUATION"):
                equation = line.split("EQUATION")[1].strip()
                reactants, products = equation.split(" <=> ")
                break
        if reactants and products:
            return reactants, products
    return None, None

def get_molecular_graphs(smiles_list):
    """Convert a list of SMILES strings to RDKit molecular objects."""
    mols = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            mols.append(mol)
        else:
            return None
    return mols

def get_reaction_fingerprint(reactant_mols, product_mols):
    """Calculate the reaction fingerprint."""
    reactant_fp = sum((AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048) for mol in reactant_mols), DataStructs.ExplicitBitVect(2048))
    product_fp = sum((AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048) for mol in product_mols), DataStructs.ExplicitBitVect(2048))
    reaction_fp = DataStructs.ExplicitBitVect(2048)
    DataStructs.BitVectXor(reaction_fp, reactant_fp, product_fp)
    return reaction_fp

def calculate_tanimoto(fp1, fp2):
    """Calculate the Tanimoto similarity between two fingerprints."""
    return DataStructs.TanimotoSimilarity(fp1, fp2)

def calculate_levenshtein(seq1, seq2):
    """Calculate the Levenshtein distance between two sequences."""
    alignments = pairwise2.align.globalxx(seq1, seq2)
    best_alignment = alignments[0]
    seqA, seqB, score, start, end = best_alignment
    distance = max(len(seqA), len(seqB)) - score
    return distance, seqA, seqB


# Main function to construct the dataset
def main():
    reactions = []
    with open('reaction_enzyme_data.csv', 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            reactions.append(row)

    reaction_data = []

    # Process each reaction
    for reaction in reactions:
        reaction_id = reaction.get('reaction_id')
        enzyme_sequence = reaction.get('enzyme_sequence')

        if not all([reaction_id, enzyme_sequence]):
            continue

        reactant_smiles, product_smiles = get_kegg_reaction_smiles(reaction_id)
        if not reactant_smiles or not product_smiles:
            continue

        reactant_mols = get_molecular_graphs(reactant_smiles.split(' + '))
        product_mols = get_molecular_graphs(product_smiles.split(' + '))

        if None in reactant_mols + product_mols:
            continue

        reaction_fp = get_reaction_fingerprint(reactant_mols, product_mols)
        reaction_data.append({
            'reaction_id': reaction_id,
            'reactant_smiles': reactant_smiles,
            'product_smiles': product_smiles,
            'reaction_fp': reaction_fp,
            'enzyme_sequence': enzyme_sequence
        })

    # Compare reactions
    dataset_rows = []
    for reacA, reacB in itertools.combinations(reaction_data, 2):
        if reacA['reaction_id'] == reacB['reaction_id']:
            continue
        tanimoto = calculate_tanimoto(reacA['reaction_fp'], reacB['reaction_fp'])
        if tanimoto >= THRESHOLD_TANIMOTO:
            enzA_seq = reacA['enzyme_sequence']
            enzB_seq = reacB['enzyme_sequence']
            if len(enzA_seq) < SEQUENCE_LENGTH_THRESHOLD and len(enzB_seq) < SEQUENCE_LENGTH_THRESHOLD:
                distance, aligned_enzA, aligned_enzB = calculate_levenshtein(enzA_seq, enzB_seq)
                if distance <= LEVENSHTEIN_THRESHOLD:
                    dataset_rows.append({
                        'reactionA_id': reacA['reaction_id'],
                        'reactionB_id': reacB['reaction_id'],
                        'reactant_smiles_A': reacA['reactant_smiles'],
                        'product_smiles_A': reacA['product_smiles'],
                        'reactant_smiles_B': reacB['reactant_smiles'],
                        'product_smiles_B': reacB['product_smiles'],
                        'reaction_fingerprint_A': reacA['reaction_fp'].ToBitString(),
                        'reaction_fingerprint_B': reacB['reaction_fp'].ToBitString(),
                        'enzymeA_sequence': enzA_seq,
                        'enzymeB_sequence': enzB_seq,
                        'aligned_enzymeA': aligned_enzA,
                        'aligned_enzymeB': aligned_enzB
                    })

    # Write dataset to CSV file
    with open('dataset.csv', 'w', newline='') as csvfile:
        fieldnames = [
            'reactionA_id', 'reactionB_id',
            'reactant_smiles_A', 'product_smiles_A',
            'reactant_smiles_B', 'product_smiles_B',
            'reaction_fingerprint_A', 'reaction_fingerprint_B',
            'enzymeA_sequence', 'enzymeB_sequence',
            'aligned_enzymeA', 'aligned_enzymeB'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in dataset_rows:
            writer.writerow(row)
    

if __name__ == "__main__":
    main()
