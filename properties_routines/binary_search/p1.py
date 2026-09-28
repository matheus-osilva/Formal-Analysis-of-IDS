from fractions import Fraction
from pathlib import Path
from utils import get_max_var_from_text, run_lukasol



def solve_for_boundary(search_type, origin_file, output_filename, folder_name):
    low = 0.0
    high = 1
    tolerance = 0.001    
    
    best_sat_val = 0.0 if search_type == 'max' else 1.0
    
    curr_low = low
    curr_high = high
    valid_search = True 
    while True:
        if (curr_high - curr_low) < tolerance:
            break
            
        mid = (curr_low + curr_high) / 2
        
        # REDUCED PRECISION: Limit denominator to 64 to avoid massive clauses
        #frac = Fraction(mid).limit_denominator(64)
        frac = Fraction(mid)
        a, b = frac.numerator, frac.denominator
        if a == 1 and b == 2:
            a = 2
            b = 4
        
        if search_type == 'max': build_p1_file_max(a, b, origin_file, output_filename, folder_name)
        elif search_type == 'min': build_p1_file_min(a, b, origin_file, output_filename, folder_name)

        is_sat = run_lukasol(f"./properties/binary_search/{folder_name}/prop1{output_filename}_{a}_{b}_{search_type}.limodsat")
        
        if is_sat is None:
            print(f"    ! Aborting search at {mid} ({a}/{b}) due to solver error.")
            valid_search = False
            break
        
        print(f"Result for {output_filename} for a={a} and b={b} in operation={search_type}: {is_sat}")
        if search_type == 'max':
            if is_sat:
                best_sat_val = mid
                curr_low = mid
            else:
                curr_high = mid
        else: # min
            if is_sat:
                best_sat_val = mid
                curr_high = mid
            else:
                curr_low = mid
                
    if not valid_search:
        return -1.0 # Indicator of failure
        
    return best_sat_val

def build_p1_file_max(a, b, origin_file, output_filename, folder_name):

    if b < 1:
        raise ValueError("Parameter 'b' must be greater than or equal to 1.")
    
    
    output_folder = Path(f'./properties/binary_search/{folder_name}/')


    try:
        # 1. Read Content & Calculate Max Var
        content = Path(origin_file).read_text(encoding='utf-8')
        
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
        # Added .limodsat extension so the file is usable
        new_filename = f"prop1{output_filename}_{a}_{b}_max.limodsat"
        
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
        print(f"Error processing {origin_file}: {e}")

def build_p1_file_min(a, b, origin_file, output_filename, folder_name):

    if b < 1:
        raise ValueError("Parameter 'b' must be greater than or equal to 1.")
    
    
    output_folder = Path(f'./properties/binary_search/{folder_name}/')


    try:
        # 1. Read Content & Calculate Max Var
        content = Path(origin_file).read_text(encoding='utf-8')
        
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
        # Added .limodsat extension so the file is usable
        new_filename = f"prop1{output_filename}_{a}_{b}_min.limodsat"
        
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
                    f_out.write(f"Unit 3 :: Implication :: 1 2\n\n")
            
            # --- MIDDLE BLOCKS: Copy the rest (old Set Phi) ---
            # Iterate from the second block onwards
            for block in blocks[1:]:
                clean_block = block.strip()
                if clean_block:
                    f_out.write("f:\n")
                    f_out.write(clean_block + "\n\n")
                    
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
        print(f"Error processing {origin_file}: {e}")

def build_p1_file_0(origin_file, output_filename, folder_name):
    
    
    output_folder = Path(f'./properties/binary_search/{folder_name}/')


    try:
        # 1. Read Content & Calculate Max Var
        content = Path(origin_file).read_text(encoding='utf-8')
        
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
        # Added .limodsat extension so the file is usable
        new_filename = f"prop1{output_filename}_0_0.limodsat"
        
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
                    # Generate the string with 'new_var'
                    str_repeated_a = " ".join([str(new_var)] * 2)
                    
                    f_out.write("f:\n")
                    f_out.write(f"{unit1_line}\n")
                    f_out.write(f"Unit 2 :: Clause      :: {str_repeated_a}\n")
                    f_out.write(f"Unit 3 :: Negation    :: 2\n")
                    f_out.write(f"Unit 4 :: Conjunction :: 2 3\n")
                    f_out.write(f"Unit 5 :: Implication :: 1 4\n\n")
            
            # --- MIDDLE BLOCKS: Copy the rest (old Set Phi) ---
            # Iterate from the second block onwards
            for block in blocks[1:]:
                clean_block = block.strip()
                if clean_block:
                    f_out.write("f:\n")
                    f_out.write(clean_block + "\n\n")
                    
            # Unit 2 has 'new_var' repeated (b-1) times
            # count_b = 9
            # str_repeated_b = " ".join([str(new_var)] * count_b)
            
            # f_out.write("f:\n")
            # f_out.write(f"Unit 1 :: Clause      :: {new_var}\n")
            # f_out.write(f"Unit 2 :: Clause      :: {str_repeated_b}\n")
            # f_out.write(f"Unit 3 :: Negation    :: 2\n")
            # f_out.write(f"Unit 4 :: Equivalence :: 1 3\n")

        print(f"Created: {output_path}")

    except Exception as e:
        print(f"Error processing {origin_file}: {e}")

def binary_search(nn, neuron):
    GLB = None
    LUB = None
    origin_file = f'./limodsat/{nn}/{neuron}.limodsat'
    output_filename = neuron
    folder_name = 'wbl_validation'

    # Test the infimum and the supremum of the interval, i.e., 0 and 1.
    build_p1_file_0(origin_file, output_filename, folder_name)
    p1_0_sat = run_lukasol(f"./properties/binary_search/{folder_name}/prop1{neuron}_0_0.limodsat")
    build_p1_file_max(2, 2, origin_file, output_filename, folder_name)
    p1_1_sat = run_lukasol(f"./properties/binary_search/{folder_name}/prop1{neuron}_2_2_max.limodsat")

    if p1_0_sat: GLB = 0
    if p1_1_sat: LUB = 1

    if GLB is None: 
        GLB = solve_for_boundary('min', origin_file, output_filename, folder_name)
    if LUB is None:
        LUB = solve_for_boundary('max', origin_file, output_filename, folder_name)

    print(f"The result for {output_filename} is [{GLB}, {LUB}].")
    return (GLB, LUB)
    



if __name__ == "__main__":
    output =[]
    for i in range(6, 15):
        result = binary_search('nn_1', f'nn_1_{i}')
        output.append(result)
        print(result)
        print('---------------------------------------')
    print(output)