import re
import subprocess
import sys
import shutil
from pathlib import Path
from fractions import Fraction

def prop1(a, b):
    # Ensure b is at least 1 to avoid negative repetition counts
    if b < 1:
        raise ValueError("Parameter 'b' must be greater than or equal to 1.")

    folder_list = ['./limodsat/nn_0', './limodsat/nn_1']
    j = -1
    
    for folder_str in folder_list:
        folder_path = Path(folder_str)
        j += 1

        # FIX 1: Wrap the string in Path() so the '/' operator works later
        output_folder = Path(f'./properties/nn_{j}/')
        
        # FIX 2: Ensure this directory exists, otherwise open() will fail
        output_folder.mkdir(parents=True, exist_ok=True)

        all_files = [
            fil for fil in folder_path.glob('*.limodsat')
        ]

        # Iterate over all files in the folder (ignoring subdirectories)
        for i, file_path in enumerate(all_files):
            if file_path.is_dir() or file_path.name.startswith('.'): 
                continue 
            
            try:
                # 1. Read Content & Calculate Max Var
                content = file_path.read_text(encoding='utf-8')
                
                # We calculate max_var from the raw content before any modification
                raw_max = get_max_var_from_text(content)
                new_var = raw_max + 1
                
                # 2. String Replacements
                # Standardize headers to 'f:'
                content = content.replace("-= Formula phi =-", "f:")
                content = content.replace("-= MODSAT Set Phi =-", "f:")
                
                # 3. Block Parsing
                blocks = content.split('f:')
                if not blocks[0].strip():
                    blocks.pop(0) # Remove empty leading block
                
                # Define output filename
                # Recommendation: Added .limodsat extension so the file is usable
                new_filename = f"prop1_neuron{i}_{a}_{b}.limodsat"
                
                # This works now because output_folder is a Path object
                output_path = output_folder / new_filename
                
                with open(output_path, 'w', encoding='utf-8') as f_out:
                    f_out.write("Sat\n\n")
                    
                    # --- BLOCK 0: Modify the original Formula Phi ---
                    if blocks:
                        first_block = blocks[0].strip()
                        
                        # Extract the original Unit 1 line
                        unit1_line = ""
                        for line in first_block.splitlines():
                            if "Unit 1" in line:
                                unit1_line = line.strip()
                                break
                        
                        if unit1_line:
                            # Generate the string with 'new_var' repeated 'a' times
                            str_repeated_a = " ".join([str(new_var)] * a)
                            
                            f_out.write("f:\n")
                            f_out.write(f"{unit1_line}\n")
                            f_out.write(f"Unit 2 :: Clause      :: {str_repeated_a}\n")
                            f_out.write(f"Unit 3 :: Implication :: 2 1\n\n")
                    
                    # --- MIDDLE BLOCKS: Copy the rest (old Set Phi) ---
                    # Iterate from the second block onwards
                    for block in blocks[1:]:
                        clean_block = block.strip()
                        if clean_block:
                            f_out.write("f:\n")
                            f_out.write(clean_block + "\n\n")
                            
                    # --- FINAL BLOCK: Add the new Robustness Logic ---
                    # Unit 2 has 'new_var' repeated (b-1) times
                    count_b = b - 1
                    str_repeated_b = " ".join([str(new_var)] * count_b)
                    
                    f_out.write("f:\n")
                    f_out.write(f"Unit 1 :: Clause      :: {new_var}\n")
                    f_out.write(f"Unit 2 :: Clause      :: {str_repeated_b}\n")
                    f_out.write(f"Unit 3 :: Negation    :: 2\n")
                    f_out.write(f"Unit 4 :: Equivalence :: 1 3\n")

                print(f"Created: {output_path}")

            except Exception as e:
                print(f"Error processing {file_path.name}: {e}")


def prop2(total_neurons, masterfile_name, nn_x):
    root_folder = Path('./properties_routines')
    master_path = root_folder / masterfile_name
    
    max_var = calculates_max_var(masterfile_name) + 1
    
    # split('f:') creates a list where each f: is a text block
    content_master = master_path.read_text(encoding='utf-8')
    blocks = content_master.split('f:')
    
    if not blocks[0].strip():
        blocks.pop(0)

    for i in range(total_neurons):
        new_file_name = f"prop2_neuron{i}.limodsat"
        new_path = Path(f'./properties/{nn_x}') / new_file_name
        
        with open(new_path, 'w', encoding='utf-8') as f_out:
            f_out.write("Sat\n\n")            
            for k, bloco in enumerate(blocks):
                cleaned_block = bloco.strip()
                if not cleaned_block:
                    continue 

                f_out.write("f:\n")

                if k < total_neurons:
                    partes = cleaned_block.split('::')
                    original_numbers = partes[-1].strip()
                    
                    logic_type = "Equivalence" if k == i else "Implication"
                    
                    f_out.write(f"Unit 1 :: Clause      :: {original_numbers}\n")
                    f_out.write(f"Unit 2 :: Clause      :: {max_var}\n")
                    f_out.write(f"Unit 3 :: {logic_type} :: 1 2\n")
                
                else:
                    # copies the modsat set phi
                    f_out.write(cleaned_block + "\n")
                
                f_out.write("\n")
        
        print(f"Created: {new_file_name}")


def prop3(total_neurons, masterfile_name, nn_x):
    root_folder = Path('./properties_routines')
    master_path = root_folder / masterfile_name
    
    raw_max = calculates_max_var(masterfile_name)
    
    var_z = raw_max + total_neurons + 1
    
    content_master = master_path.read_text(encoding='utf-8')
    blocks = content_master.split('f:')
    
    if not blocks[0].strip():
        blocks.pop(0)

    # Loop i: Create one file for each neuron acting as the "Max" candidate
    for i in range(total_neurons):
        new_file_name = f"prop3_neuron{i}.limodsat"
        new_path = Path(f'./properties/{nn_x}') / new_file_name
        
        # Ensure directory exists (optional, but good practice)
        new_path.parent.mkdir(parents=True, exist_ok=True)

        with open(new_path, 'w', encoding='utf-8') as f_out:
            f_out.write("Cons\n\n")
            
            # --- PART 1: Iterate through blocks (Main Body) ---
            for k, block in enumerate(blocks):
                cleaned_block = block.strip()
                if not cleaned_block:
                    continue

                f_out.write("f:\n")

                if k < total_neurons:
                    # Logic: Each neuron k gets a unique variable (raw_max + k + 1)
                                        
                    parts = cleaned_block.split('::')
                    original_numbers = parts[-1].strip()
                    
                    # Variable specific to this block's neuron
                    current_neuron_var = raw_max + k + 1
                    
                    f_out.write(f"Unit 1 :: Clause      :: {original_numbers}\n")
                    f_out.write(f"Unit 2 :: Clause      :: {current_neuron_var}\n")
                    f_out.write(f"Unit 3 :: Equivalence :: 1 2\n")
                
                else:
                    # Copies extra blocks (like set phi)
                    f_out.write(cleaned_block + "\n")
                
                f_out.write("\n")

            # --- PART 2: Footer A (Transition/Negation with Z) ---
            str_z_repeated_9 = " ".join([str(var_z)] * 9)
            
            f_out.write("f:\n")
            f_out.write(f"Unit 1 :: Clause      :: {var_z}\n")
            f_out.write(f"Unit 2 :: Clause      :: {str_z_repeated_9}\n")
            f_out.write(f"Unit 3 :: Negation    :: 2\n")
            f_out.write(f"Unit 4 :: Equivalence :: 1 3\n\n")

            # --- PART 3: Footer B (Max Verification Logic) ---
            # This part changes dynamically depending on 'i' (the current file's max candidate)

            # Step 3.1: Define Clauses for ALL neurons
            for k in range(total_neurons):
                if k == i:
                    continue
                val = raw_max + k + 1
                f_out.write("f:\n")
                f_out.write(f"Unit 1 :: Clause      :: {val}\n")

                f_out.write(f"Unit 2 :: Clause      :: {raw_max + i + 1}\n")

                f_out.write(f"Unit 3 :: Implication :: 2 1\n")
                f_out.write(f"\n")
            
            
            current_unit = 1
            unit_indices = []

            f_out.write("C:\n")
            # Step 3.4: Clause Z repeated
            str_z_repeated_5 = " ".join([str(var_z)] * 5)
            idx_clause_z = current_unit
            
            f_out.write(f"Unit {current_unit} :: Clause      :: {str_z_repeated_5}\n")
            current_unit += 1

            # Step 3.1: Define Clauses for ALL neurons
            for k in range(total_neurons):
                if k == i:
                    continue
                val = raw_max + k + 1
                f_out.write(f"Unit {current_unit} :: Clause      :: {val}\n")
                current_unit += 1
                f_out.write(f"Unit {current_unit} :: Implication :: {current_unit-1} {idx_clause_z}\n")
                unit_indices.append(current_unit) # Store index for later reference
                current_unit += 1

            # Step 3.3: Minimum List
            # The list contains all units EXCEPT the target one
            others_indices = [str(idx) for k, idx in enumerate(unit_indices)]
            str_others = " ".join(others_indices)
            
            f_out.write(f"Unit {current_unit} :: Minimum     :: {str_others}\n")
            current_unit += 1

        print(f"Created: {new_file_name}")


def prop3_contrapositive(total_neurons, masterfile_name, nn_x):
    root_folder = Path('./properties_routines')
    master_path = root_folder / masterfile_name
    
    raw_max = calculates_max_var(masterfile_name)
    
    var_z = raw_max + total_neurons + 1
    
    content_master = master_path.read_text(encoding='utf-8')
    blocks = content_master.split('f:')
    
    if not blocks[0].strip():
        blocks.pop(0)

    # Loop i: Create one file for each neuron acting as the "Max" candidate
    for i in range(total_neurons):
        new_file_name = f"prop3_neuron{i}.limodsat"
        new_path = Path(f'./properties/{nn_x}') / new_file_name
        
        # Ensure directory exists (optional, but good practice)
        new_path.parent.mkdir(parents=True, exist_ok=True)

        with open(new_path, 'w', encoding='utf-8') as f_out:
            f_out.write("Sat\n\n")
            
            # --- PART 1: Iterate through blocks (Main Body) ---
            for k, block in enumerate(blocks):
                cleaned_block = block.strip()
                if not cleaned_block:
                    continue

                f_out.write("f:\n")

                if k < total_neurons:
                    # Logic: Each neuron k gets a unique variable (raw_max + k + 1)
                                        
                    parts = cleaned_block.split('::')
                    original_numbers = parts[-1].strip()
                    
                    # Variable specific to this block's neuron
                    current_neuron_var = raw_max + k + 1
                    
                    f_out.write(f"Unit 1 :: Clause      :: {original_numbers}\n")
                    f_out.write(f"Unit 2 :: Clause      :: {current_neuron_var}\n")
                    f_out.write(f"Unit 3 :: Equivalence :: 1 2\n")
                
                else:
                    # Copies extra blocks (like set phi)
                    f_out.write(cleaned_block + "\n")
                
                f_out.write("\n")

            # --- PART 2: Footer A (Transition/Negation with Z) ---
            str_z_repeated_9 = " ".join([str(var_z)] * 9)
            
            f_out.write("f:\n")
            f_out.write(f"Unit 1 :: Clause      :: {var_z}\n")
            f_out.write(f"Unit 2 :: Clause      :: {str_z_repeated_9}\n")
            f_out.write(f"Unit 3 :: Negation    :: 2\n")
            f_out.write(f"Unit 4 :: Equivalence :: 1 3\n\n")

            # --- PART 3: Footer B (Max Verification Logic) ---
            # This part changes dynamically depending on 'i' (the current file's max candidate)

            # Step 3.1: Define Clauses for ALL neurons
            for k in range(total_neurons):
                if k == i:
                    continue
                val = raw_max + k + 1
                f_out.write("f:\n")
                f_out.write(f"Unit 1 :: Clause      :: {val}\n")

                f_out.write(f"Unit 2 :: Clause      :: {raw_max + i + 1}\n")

                f_out.write(f"Unit 3 :: Implication :: 2 1\n")
                f_out.write(f"\n")
            
            
            current_unit = 1
            unit_indices = []

            f_out.write("f:\n")
            # Step 3.1: Define Clauses for ALL neurons
            for k in range(total_neurons):
                val = raw_max + k + 1
                f_out.write(f"Unit {current_unit} :: Clause      :: {val}\n")
                unit_indices.append(current_unit) # Store index for later reference
                current_unit += 1

            # Step 3.3: Maximum List
            # The list contains all units EXCEPT the target one
            others_indices = [str(idx) for k, idx in enumerate(unit_indices) if k != i]
            str_others = " ".join(others_indices)
            
            idx_max_unit = current_unit
            f_out.write(f"Unit {current_unit} :: Maximum     :: {str_others}\n")
            current_unit += 1

            # Step 3.4: Clause Z repeated and Final Implication
            str_z_repeated_5 = " ".join([str(var_z)] * 5)
            idx_clause_z = current_unit
            
            f_out.write(f"Unit {current_unit} :: Clause      :: {str_z_repeated_5}\n")
            current_unit += 1
            
            f_out.write(f"Unit {current_unit} :: Implication :: {idx_clause_z} {idx_max_unit}\n")

        print(f"Created: {new_file_name}")


def prop4(total_neurons, masterfile_name, nn_x, a, b):
    root_folder = Path('./properties_routines')
    master_path = root_folder / masterfile_name
    
    raw_max = calculates_max_var(masterfile_name)
    
    var_z = raw_max + total_neurons + 1
    
    content_master = master_path.read_text(encoding='utf-8')
    blocks = content_master.split('f:')
    
    if not blocks[0].strip():
        blocks.pop(0)

    # Loop i: Create one file for each neuron acting as the "Max" candidate
    for i in range(total_neurons):
        new_file_name = f"prop4_neuron{i}_{a}_{b}.limodsat"
        new_path = Path(f'./properties/{nn_x}') / new_file_name
        
        # Ensure directory exists (optional, but good practice)
        new_path.parent.mkdir(parents=True, exist_ok=True)

        with open(new_path, 'w', encoding='utf-8') as f_out:
            f_out.write("Cons\n\n")
            
            # --- PART 1: Iterate through blocks (Main Body) ---
            for k, block in enumerate(blocks):
                cleaned_block = block.strip()
                if not cleaned_block:
                    continue

                f_out.write("f:\n")

                if k < total_neurons:
                    # Logic: Each neuron k gets a unique variable (raw_max + k + 1)
                    
                    parts = cleaned_block.split('::')
                    original_numbers = parts[-1].strip()
                    
                    # Variable specific to this block's neuron
                    current_neuron_var = raw_max + k + 1
                    
                    f_out.write(f"Unit 1 :: Clause      :: {original_numbers}\n")
                    f_out.write(f"Unit 2 :: Clause      :: {current_neuron_var}\n")
                    f_out.write(f"Unit 3 :: Equivalence :: 1 2\n")
                
                else:
                    # Copies extra blocks (like set phi)
                    f_out.write(cleaned_block + "\n")
                
                f_out.write("\n")

            # --- PART 2: Footer A (Transition/Negation with Z) ---
            str_z_repeated_b = " ".join([str(var_z)] * (b-1))
            
            f_out.write("f:\n")
            f_out.write(f"Unit 1 :: Clause      :: {var_z}\n")
            f_out.write(f"Unit 2 :: Clause      :: {str_z_repeated_b}\n")
            f_out.write(f"Unit 3 :: Negation    :: 2\n")
            f_out.write(f"Unit 4 :: Equivalence :: 1 3\n\n")

            # --- PART 3: Footer B (Max Verification Logic) ---
            # This part changes dynamically depending on 'i' (the current file's max candidate)

            # Step 3.1: Define Clauses for ALL neurons
            for k in range(total_neurons):
                if k == i:
                    continue
                val = raw_max + k + 1
                f_out.write("f:\n")
                f_out.write(f"Unit 1 :: Clause      :: {val}\n")

                f_out.write(f"Unit 2 :: Clause      :: {raw_max + i + 1}\n")

                f_out.write(f"Unit 3 :: Implication :: 2 1\n")
                f_out.write(f"\n")
            
            current_unit = 1
            unit_indices = []
            f_out.write("C:\n")
            # Step 3.1: Clause Z repeated
            str_z_repeated_a = " ".join([str(var_z)] * a)
            idx_clause_z = current_unit
            f_out.write(f"Unit {current_unit} :: Clause      :: {str_z_repeated_a}\n")
            current_unit += 1
            f_out.write(f"Unit {current_unit} :: Clause      :: {raw_max + i + 1}\n")
            idx_target = current_unit
            current_unit += 1
            # Step 3.2: Define Clauses for ALL neurons
            for k in range(total_neurons):
                if k == i:
                    continue
                val = raw_max + k + 1
                f_out.write(f"Unit {current_unit} :: Clause      :: {val}\n")
                current_unit += 1
                f_out.write(f"Unit {current_unit} :: Implication :: {idx_target} {current_unit - 1}\n")
                current_unit += 1
                f_out.write(f"Unit {current_unit} :: Negation    :: {current_unit - 1}\n")
                current_unit += 1
                f_out.write(f"Unit {current_unit} :: Implication :: {idx_clause_z} {current_unit - 1}\n")
                unit_indices.append(current_unit) # Store index for later reference
                current_unit += 1
            

            # Step 3.3: Minimum List
            # The list contains all units EXCEPT the target one
            others_indices = [str(idx) for k, idx in enumerate(unit_indices)]
            str_others = " ".join(others_indices)
            
            f_out.write(f"Unit {current_unit} :: Minimum     :: {str_others}\n")

        print(f"Created: {new_file_name}")


def prop4_contrapositive(total_neurons, masterfile_name, nn_x, a, b):
    root_folder = Path('./properties_routines')
    master_path = root_folder / masterfile_name
    
    raw_max = calculates_max_var(masterfile_name)
    
    var_z = raw_max + total_neurons + 1
    
    content_master = master_path.read_text(encoding='utf-8')
    blocks = content_master.split('f:')
    
    if not blocks[0].strip():
        blocks.pop(0)

    # Loop i: Create one file for each neuron acting as the "Max" candidate
    for i in range(total_neurons):
        new_file_name = f"prop4_neuron{i}_{a}_{b}.limodsat"
        new_path = Path(f'./properties/{nn_x}') / new_file_name
        
        # Ensure directory exists (optional, but good practice)
        new_path.parent.mkdir(parents=True, exist_ok=True)

        with open(new_path, 'w', encoding='utf-8') as f_out:
            f_out.write("Sat\n\n")
            
            # --- PART 1: Iterate through blocks (Main Body) ---
            for k, block in enumerate(blocks):
                cleaned_block = block.strip()
                if not cleaned_block:
                    continue

                f_out.write("f:\n")

                if k < total_neurons:
                    # Logic: Each neuron k gets a unique variable (raw_max + k + 1)
                    
                    parts = cleaned_block.split('::')
                    original_numbers = parts[-1].strip()
                    
                    # Variable specific to this block's neuron
                    current_neuron_var = raw_max + k + 1
                    
                    f_out.write(f"Unit 1 :: Clause      :: {original_numbers}\n")
                    f_out.write(f"Unit 2 :: Clause      :: {current_neuron_var}\n")
                    f_out.write(f"Unit 3 :: Equivalence :: 1 2\n")
                
                else:
                    # Copies extra blocks (like set phi)
                    f_out.write(cleaned_block + "\n")
                
                f_out.write("\n")

            # --- PART 2: Footer A (Transition/Negation with Z) ---
            str_z_repeated_b = " ".join([str(var_z)] * (b-1))
            
            f_out.write("f:\n")
            f_out.write(f"Unit 1 :: Clause      :: {var_z}\n")
            f_out.write(f"Unit 2 :: Clause      :: {str_z_repeated_b}\n")
            f_out.write(f"Unit 3 :: Negation    :: 2\n")
            f_out.write(f"Unit 4 :: Equivalence :: 1 3\n\n")

            # --- PART 3: Footer B (Max Verification Logic) ---
            # This part changes dynamically depending on 'i' (the current file's max candidate)

            # Step 3.1: Define Clauses for ALL neurons
            for k in range(total_neurons):
                if k == i:
                    continue
                val = raw_max + k + 1
                f_out.write("f:\n")
                f_out.write(f"Unit 1 :: Clause      :: {val}\n")

                f_out.write(f"Unit 2 :: Clause      :: {raw_max + i + 1}\n")

                f_out.write(f"Unit 3 :: Implication :: 2 1\n")
                f_out.write(f"\n")
            
            current_unit = 1
            unit_indices = []
            f_out.write("f:\n")
            # Step 3.1: Define Clauses for ALL neurons
            for k in range(total_neurons):
                val = raw_max + k + 1
                f_out.write(f"Unit {current_unit} :: Clause      :: {val}\n")
                unit_indices.append(current_unit) # Store index for later reference
                current_unit += 1
            
            # Identify the unit index of our candidate 'i'
            target_unit_idx = unit_indices[i]

            # Step 3.3: Maximum List
            # The list contains all units EXCEPT the target one
            others_indices = [str(idx) for k, idx in enumerate(unit_indices) if k != i]
            str_others = " ".join(others_indices)
            
            idx_max_unit = current_unit
            f_out.write(f"Unit {current_unit} :: Maximum     :: {str_others}\n")
            current_unit += 1


            f_out.write(f"Unit {current_unit} :: Implication :: {target_unit_idx} {idx_max_unit}\n")
            current_unit += 1

            idx_difference_max_target = current_unit
            f_out.write(f"Unit {current_unit} :: Negation    :: {current_unit - 1}\n")
            current_unit += 1
            
            # Step 3.4: Clause Z repeated and Final Implication
            str_z_repeated_a = " ".join([str(var_z)] * a)
            idx_clause_z = current_unit
            f_out.write(f"Unit {current_unit} :: Clause      :: {str_z_repeated_a}\n")
            current_unit += 1
            f_out.write(f"Unit {current_unit} :: Implication :: {idx_difference_max_target} {idx_clause_z} \n")

        print(f"Created: {new_file_name}")


# --- Funções Utilitárias para o Solver ---

def find_lukasol_binary():
    """
    Tenta localizar o binário 'lukasol' dinamicamente.
    Procura no diretório atual e em diretórios pais/filhos comuns.
    """
    filename = 'lukasol'
    if sys.platform.startswith('win'):
        filename += '.exe'
    
    # 1. Tentar encontrar no PATH do sistema
    path_in_env = shutil.which(filename)
    if path_in_env:
        return Path(path_in_env)

    # 2. Procurar recursivamente a partir do diretório atual e subir alguns níveis
    search_root = Path.cwd()
    
    # Tenta subir até 3 níveis para achar a pasta raiz do projeto caso este script esteja em subpasta
    for _ in range(4): 
        # Procura em ./bin/Release ou qualquer subpasta bin
        found = list(search_root.rglob(f"**/{filename}"))
        if found:
            # Retorna o primeiro encontrado (preferencialmente Release)
            # Prioriza caminhos que contenham 'Release'
            release_bins = [p for p in found if 'Release' in str(p)]
            return release_bins[0] if release_bins else found[0]
        
        if search_root.parent == search_root: # Chegou na raiz do sistema
            break
        search_root = search_root.parent

    raise FileNotFoundError("O binário 'lukasol' não foi encontrado. Verifique se ele foi compilado.")

def run_lukasol(file_path, solver_path):
    """
    Executa o lukasol para o arquivo dado e retorna True se SAT, False se unSAT.
    """
    try:
        # Executa o comando: ./lukasol -m arquivo.limodsat
        result = subprocess.run(
            [str(solver_path), '-mip', str(file_path)],
            capture_output=True,
            text=True,
            check=True
        )
        
        output = result.stdout
        
        # Verifica a saída padrão conforme seu exemplo
        if "Analysing satisfiability... SAT" in output:
            return True # Satisfiável
        elif "Analysing satisfiability... unSAT" in output:
            return False # Insatisfiável
        else:
            print(f"Aviso: Saída inesperada para {file_path.name}")
            return False

    except subprocess.CalledProcessError as e:
        print(f"Erro ao executar lukasol: {e}")
        return False

# --- Funções de Manipulação de Arquivos ---

def calculates_max_var(masterfile_name):
    # Ajuste de caminho para robustez
    file_path = Path('./properties_routines') / masterfile_name
    try:
        content = file_path.read_text(encoding='utf-8')
        all_numbers = [int(num) for num in re.findall(r'\d+', content)]
        return max(all_numbers) if all_numbers else 0
    except FileNotFoundError:
        print(f"Error: File '{masterfile_name}' was not found.")
        return 0

def get_max_var_from_text(content):
    all_numbers = [int(num) for num in re.findall(r'\d+', content)]
    return max(all_numbers) if all_numbers else 0

def build_masterfile():
    folder_list = ['./limodsat/nn_0', './limodsat/nn_1']
    j = -1
    for folder in folder_list:
        j+=1
        path = Path(folder)
        output_filename = f"nn_{j}_master.limodsat"
        string_key1 = "-= Formula phi =-"
        string_key2 = "f:"
        string_key3 = "-= MODSAT Set Phi =-"

        # Garante que a pasta de output existe
        Path("./properties_routines").mkdir(exist_ok=True)

        all_files = list(path.glob('*.limodsat'))
        total_files = len(all_files)

        with open(Path("./properties_routines") / output_filename, 'w', encoding='utf-8') as output_file:
            for i, file_path in enumerate(all_files):
                try:
                    content = file_path.read_text(encoding='utf-8')
                    content = content.replace(string_key1, string_key2)
                    if string_key3 in content:
                        if i == total_files -1:
                            content = content.replace(string_key3, "")
                        else:
                            content = content.split(string_key3)[0]
                    output_file.write(content)
                except Exception as e:
                    print(f"Error in {file_path.name}: {e}")

# --- Geração Unitária e Busca Binária ---

def generate_prop1_single(file_path_in, output_path, a, b):
    """
    Gera um ÚNICO arquivo de propriedade 1 baseado em um arquivo de entrada.
    Retorna True se sucesso, False caso contrário.
    """
    if b < 1: return False
    
    try:
        content = file_path_in.read_text(encoding='utf-8')
        raw_max = get_max_var_from_text(content)
        new_var = raw_max + 1
        
        content = content.replace("-= Formula phi =-", "f:")
        content = content.replace("-= MODSAT Set Phi =-", "f:")
        
        blocks = content.split('f:')
        if not blocks[0].strip():
            blocks.pop(0)
            
        with open(output_path, 'w', encoding='utf-8') as f_out:
            f_out.write("Sat\n\n")
            
            # Bloco 0
            if blocks:
                first_block = blocks[0].strip()
                unit1_line = ""
                for line in first_block.splitlines():
                    if "Unit 1" in line:
                        unit1_line = line.strip()
                        break
                
                if unit1_line:
                    str_repeated_a = " ".join([str(new_var)] * a)
                    f_out.write("f:\n")
                    f_out.write(f"{unit1_line}\n")
                    f_out.write(f"Unit 2 :: Clause      :: {str_repeated_a}\n")
                    f_out.write(f"Unit 3 :: Implication :: 2 1\n\n")
            
            # Blocos do meio
            for block in blocks[1:]:
                clean_block = block.strip()
                if clean_block:
                    f_out.write("f:\n")
                    f_out.write(clean_block + "\n\n")
            
            # Bloco Final
            count_b = b - 1
            # Evitar erro se b=1 (count_b=0), join lida bem com lista vazia
            str_repeated_b = " ".join([str(new_var)] * count_b)
            
            f_out.write("f:\n")
            f_out.write(f"Unit 1 :: Clause      :: {new_var}\n")
            f_out.write(f"Unit 2 :: Clause      :: {str_repeated_b}\n")
            f_out.write(f"Unit 3 :: Negation    :: 2\n")
            f_out.write(f"Unit 4 :: Equivalence :: 1 3\n")
            
        return True
    except Exception as e:
        print(f"Erro gerando arquivo temp: {e}")
        return False

def binary_search_prop1():
    print("--- Iniciando Busca Binária para Propriedade 1 ---")
    
    try:
        solver_path = find_lukasol_binary()
        print(f"Solver encontrado em: {solver_path}")
    except FileNotFoundError as e:
        print(e)
        return

    # Definir quais redes neurais queremos testar
    # Exemplo: nn_0 e nn_1
    target_folders = ['./limodsat/nn_1', './limodsat/nn_0']

    for folder_str in target_folders:
        folder_path = Path(folder_str)
        if not folder_path.exists():
            continue
            
        nn_name = folder_path.name # ex: nn_0
        print(f"\nProcessando Rede: {nn_name}")

        # Listar os neurônios (arquivos .limodsat originais)
        neuron_files = [f for f in folder_path.glob('*.limodsat') 
                        if not f.name.startswith('.')]
        
        for n_file in neuron_files:
            print(f"  > Neurônio: {n_file.name}")
            
            # Parâmetros da Busca Binária
            low = 0.0
            high = 1.0
            tolerance = 0.01 # Precisão desejada (1%)
            iterations = 0
            max_iterations = 20 # Evitar loop infinito
            
            # Pasta temporária para arquivos de teste
            temp_dir = Path(f'./temp_search/{nn_name}')
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            best_sat_val = 0.0 # Guarda o maior valor que deu SAT (ou menor, dependendo da lógica)
            
            while (high - low) > tolerance and iterations < max_iterations:
                iterations += 1
                mid = (low + high) / 2
                
                # Converter float 'mid' para fração a/b
                # limit_denominator(100) garante que b não fique gigante, 
                # mantendo os arquivos legíveis e próximos da lógica original.
                frac = Fraction(mid).limit_denominator(1000) 
                a, b = frac.numerator, frac.denominator
                
                # Nome do arquivo temporário
                temp_file = temp_dir / f"temp_{n_file.stem}_{a}_{b}.limodsat"
                
                # 1. Gerar o arquivo com a constante a/b
                generate_prop1_single(n_file, temp_file, a, b)
                
                # 2. Rodar o Solver
                is_sat = run_lukasol(temp_file, solver_path)
                
                # 3. Decidir direção da busca
                # LÓGICA:
                # Se SAT -> A propriedade é satisfeita com essa constante.
                # Dependendo do objetivo: 
                # Se queremos o MAIOR valor possível (ex: raio de robustez):
                #    SAT significa "ainda aguenta", tentamos subir (low = mid).
                # Se SAT significa "falha encontrada" (ex: achou um contra-exemplo),
                #    então para ser robusto queremos UNSAT.
                
                # ASSUMINDO: SAT = Propriedade Satisfeita. Queremos encontrar o limiar superior.
                if is_sat:
                    best_sat_val = mid
                    low = mid # Tenta aumentar a constante
                    print(f"    Iter {iterations}: {mid:.4f} ({a}/{b}) -> SAT (Subindo)")
                else:
                    high = mid # Constante muito alta, diminui
                    print(f"    Iter {iterations}: {mid:.4f} ({a}/{b}) -> unSAT (Descendo)")
            
            # Limpeza (Opcional): remover arquivos temporários
            # shutil.rmtree(temp_dir) 
            
            print(f"    Resultado Final: Constante ~= {best_sat_val:.4f} (Intervalo: {low:.4f} - {high:.4f})")


# --- Execução Principal ---

if __name__ == "__main__":
    # build_masterfile()
    binary_search_prop1()
    