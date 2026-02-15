
import os
import h5py
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def load_one_file(path, label):
    try:
        with h5py.File(path, 'r') as f:
            data = f['vibration_data'][:]
            return data, label
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None

def consolidate_folder(root_dir, machine_id, output_dir):
    print(f"\nProcessing {machine_id}...")
    
    if machine_id == "test":
        base_path = "new_test"
    else:
        base_path = os.path.join("new_train", machine_id)
        
    good_path = os.path.join(base_path, "good")
    bad_path = os.path.join(base_path, "bad")
    
    good_files = [os.path.join(good_path, f) for f in os.listdir(good_path) if f.endswith('.h5')] if os.path.exists(good_path) else []
    bad_files = [os.path.join(bad_path, f) for f in os.listdir(bad_path) if f.endswith('.h5')] if os.path.exists(bad_path) else []
    
    all_files = [(f, 1) for f in good_files] + [(f, 0) for f in bad_files]
    
    if not all_files:
        print(f"No files found for {machine_id}")
        return

    print(f"Loading {len(all_files)} files...")
    X, y = [], []
    
    # Use threads to speed up reading before consolidation
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(load_one_file, f, l) for f, l in all_files]
        for future in tqdm(futures, desc=f"Consolidating {machine_id}"):
            res = future.result()
            if res:
                X.append(res[0])
                y.append(res[1])
    
    if not X:
        return
        
    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.int32)
    
    output_path = os.path.join(output_dir, f"{machine_id}_consolidated.h5")
    print(f"Saving to {output_path} (Shape: {X.shape})...")
    
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('vibration_data', data=X, compression="gzip")
        f.create_dataset('label', data=y)
    
    print(f"Successfully consolidated {machine_id}")

def main():
    output_dir = "data_consolidated"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    machines = ["M01", "M02", "M03", "test"]
    for m in machines:
        consolidate_folder("new_train", m, output_dir)

if __name__ == "__main__":
    main()
