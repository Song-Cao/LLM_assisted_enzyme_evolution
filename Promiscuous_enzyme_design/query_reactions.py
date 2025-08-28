import requests
import csv
import time

def query_kegg_reaction(reaction_id):
    """
    Query KEGG database for a given reaction ID.
    """
    url = f"http://rest.kegg.jp/get/{reaction_id}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        print(f"Failed to retrieve data for {reaction_id}")
        return None

def get_reactions_from_kegg():
    """
    Retrieve all reaction IDs from KEGG database.
    """
    url = "http://rest.kegg.jp/list/reaction"
    response = requests.get(url)
    if response.status_code == 200:
        reactions = response.text.strip().split('\n')
        return [reaction.split('\t')[0] for reaction in reactions]
    else:
        print("Failed to retrieve reaction list from KEGG")
        return []

def extract_enzyme_ids(kegg_entry):
    """
    Extract enzyme IDs (EC numbers) from the KEGG entry for a reaction.
    """
    enzyme_ids = []
    lines = kegg_entry.splitlines()
    for line in lines:
        if line.startswith('ENZYME'):
            enzyme_ids.extend(line.split()[1:])
    return enzyme_ids

def get_uniprot_ids_from_ec(ec_number):
    """
    Retrieve UniProt IDs corresponding to a given EC number.
    """
    url = f"https://www.uniprot.org/uniprot/?query=ec:{ec_number}&format=tab&columns=id"
    response = requests.get(url)
    if response.status_code == 200:
        uniprot_ids = response.text.strip().split('\n')[1:]  # Skip header
        return uniprot_ids
    else:
        print(f"Failed to retrieve UniProt IDs for EC number {ec_number}")
        return []

def get_enzyme_sequence_from_uniprot(uniprot_id):
    """
    Retrieve enzyme sequence from UniProt using UniProt ID.
    """
    url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
    response = requests.get(url)
    if response.status_code == 200:
        fasta_data = response.text
        sequence = ''.join(fasta_data.split('\n')[1:])  # Remove header and join sequence lines
        return sequence
    else:
        print(f"Failed to retrieve sequence for UniProt ID {uniprot_id}")
        return None

def main():
    reactions = get_reactions_from_kegg()
    reaction_enzyme_data = []

    for reaction in reactions:
        kegg_entry = query_kegg_reaction(reaction)
        if kegg_entry:
            enzyme_ids = extract_enzyme_ids(kegg_entry)
            for enzyme_id in enzyme_ids:
                uniprot_ids = get_uniprot_ids_from_ec(enzyme_id)
                for uniprot_id in uniprot_ids:
                    enzyme_sequence = get_enzyme_sequence_from_uniprot(uniprot_id)
                    if enzyme_sequence:
                        reaction_enzyme_data.append({
                            'reaction_id': reaction,
                            'enzyme_id': enzyme_id,
                            'uniprot_id': uniprot_id,
                            'enzyme_sequence': enzyme_sequence
                        })
                        # To avoid overwhelming the server
                        time.sleep(0.2)

    # Write results to CSV
    with open('reaction_enzyme_data.csv', 'w', newline='') as csvfile:
        fieldnames = ['reaction_id', 'enzyme_id', 'uniprot_id', 'enzyme_sequence']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in reaction_enzyme_data:
            writer.writerow(row)

if __name__ == "__main__":
    main()
