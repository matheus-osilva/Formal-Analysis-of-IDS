from pathlib import Path
import re
import shutil
import subprocess
import sys



def get_max_var_from_text(content):
    all_numbers = [int(num) for num in re.findall(r'\d+', content)]
    return max(all_numbers) if all_numbers else 0

def find_lukasol_binary():
    filename = 'lukasol'
    if sys.platform.startswith('win'):
        filename += '.exe'
    
    path_in_env = shutil.which(filename)
    if path_in_env:
        return Path(path_in_env)

    search_root = Path.cwd()
    for _ in range(4): 
        found = list(search_root.rglob(f"**/{filename}"))
        if found:
            release_bins = [p for p in found if 'Release' in str(p)]
            return release_bins[0] if release_bins else found[0]
        if search_root.parent == search_root:
            break
        search_root = search_root.parent

def run_lukasol(file_path):
    """
    Returns:
        True: SAT for satisfiability test or VALID for consequence test
        False: unSAT or INvalid
        None: Error/Unknown (CRITICAL CHANGE)
    """
    try:
        solver_path = find_lukasol_binary()
    except FileNotFoundError as e:
        print(e)
        return
    
    try:
        result = subprocess.run(
            [str(solver_path), '-mip', str(file_path)],
            capture_output=True,
            text=True,
            check=True
        )
        output = result.stdout
        
        if "Analysing satisfiability... SAT" in output or "Analysing consequence validity... VALID" in output:
            return True
        elif "Analysing satisfiability... unSAT" in output or "Analysing consequence validity... INvalid" in output:
            return False
        else:
            # Found output that is neither SAT nor UNSAT
            print(f"  [WARNING] Solver output unclear for {file_path.name}")
            # print(output[:200]) # Uncomment to debug
            return None 

    except subprocess.CalledProcessError as e:
        print(f"  [ERROR] Solver crashed: {e}")
        return None

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