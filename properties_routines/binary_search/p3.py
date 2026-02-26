from fractions import Fraction
from pathlib import Path
from utils import run_lukasol, calculates_max_var
import re



def solve(output_filename, folder_name, masterfile_name, neuron_number, nn):
    low = 0.0
    high = 1.0
    tolerance = 0.01    
    
    best_sat_val = 0.0
    
    curr_low = low
    curr_high = high
    valid_search = True 
    while True:
        if (curr_high - curr_low) < tolerance:
            break
            
        mid = (curr_low + curr_high) / 2
        
        # REDUCED PRECISION: Limit denominator to 64 to avoid massive clauses
        frac = Fraction(mid).limit_denominator(64)
        a, b = frac.numerator, frac.denominator
        if a == 1 and b == 2:
            a = 2
            b = 4
        
        build_p3_file(a, b,output_filename, folder_name, masterfile_name, neuron_number, nn)

        is_sat = run_lukasol(f"./properties/binary_search/{folder_name}/prop3_{nn}_{output_filename}_{a}_{b}.limodsat")
        
        if is_sat is None:
            print(f"    ! Aborting search at {mid} ({a}/{b}) due to solver error.")
            valid_search = False
            break
        
        print(f"Result for {output_filename} for a={a} and b={b}: {is_sat}")
        if not is_sat:
            best_sat_val = mid
            curr_low = mid
        else:
            curr_high = mid
                
    if not valid_search:
        return -1.0 # Indicator of failure
        
    return best_sat_val

def build_p3_file(a, b, output_filename, folder_name, masterfile_name, neuron_number, nn):

    if b < 1:
        raise ValueError("Parameter 'b' must be greater than or equal to 1.")
    root_folder = Path('./properties_routines')
    master_path = root_folder / masterfile_name
    
    raw_max = calculates_max_var(masterfile_name)
    
    var_z = raw_max + 15 + 1
    
    content_master = master_path.read_text(encoding='utf-8')
    blocks = content_master.split('f:')
    
    if not blocks[0].strip():
        blocks.pop(0)
    
    new_file_name = f"prop3_{nn}_{output_filename}_{a}_{b}.limodsat"
    new_path = Path(f'./properties/binary_search/{folder_name}/') / new_file_name
    
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

            if k < 15:
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
        str_z_repeated = " ".join([str(var_z)] * (b-1))
        
        f_out.write("f:\n")
        f_out.write(f"Unit 1 :: Clause      :: {var_z}\n")
        f_out.write(f"Unit 2 :: Clause      :: {str_z_repeated}\n")
        f_out.write(f"Unit 3 :: Negation    :: 2\n")
        f_out.write(f"Unit 4 :: Equivalence :: 1 3\n\n")

        # --- PART 3: Footer B (Max Verification Logic) ---
        # This part changes dynamically depending on 'i' (the current file's max candidate)

        # Step 3.1: Define Clauses for ALL neurons
        for k in range(15):
            if k == neuron_number:
                continue
            val = raw_max + k + 1
            f_out.write("f:\n")
            f_out.write(f"Unit 1 :: Clause      :: {val}\n")

            f_out.write(f"Unit 2 :: Clause      :: {raw_max + neuron_number + 1}\n")

            f_out.write(f"Unit 3 :: Implication :: 2 1\n")
            f_out.write(f"\n")
        
        
        current_unit = 1
        unit_indices = []

        f_out.write("C:\n")
        # Step 3.4: Clause Z repeated
        str_z_repeated_a = " ".join([str(var_z)] * a)
        idx_clause_z = current_unit
        
        f_out.write(f"Unit {current_unit} :: Clause      :: {str_z_repeated_a}\n")
        current_unit += 1

        # Step 3.1: Define Clauses for ALL neurons
        for k in range(15):
            if k == neuron_number:
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

def binary_search(masterfile_name, neuron, nn):
    output_filename = neuron
    folder_name = 'temporary'

    result = solve(output_filename, folder_name, masterfile_name, neuron, nn)

    print(f"The result for {output_filename} is {result}.")
    return result




if __name__ == "__main__":
    output =[]
    for i in range(0, 15):
        output.append(binary_search('nn_0_master.limodsat', i, 'nn_0'))
    
    print(output)