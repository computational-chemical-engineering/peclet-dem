
import struct
import os
import sys
import numpy as np

def inspect_vti(filename):
    if not os.path.exists(filename):
        print(f"File {filename} does not exist.")
        return

    print(f"Inspecting {filename}...")
    with open(filename, "rb") as f:
        content = f.read()

    # Find AppendedData
    marker = b'<AppendedData encoding="raw">'
    start_idx = content.find(marker)
    if start_idx == -1:
        print("Could not find AppendedData section.")
        return
    
    # Payload starts after marker + newline + "_" + size(8 bytes)
    # The XML says: <AppendedData encoding="raw">\n_
    # Let's search for "_\n" or just "_"
    # The code: file << "  <AppendedData encoding=\"raw\">\n"; file << "_";
    # So it should be ...>\n_
    payload_start = content.find(b'_', start_idx) + 1
    
    # Read 8 bytes size
    size_bytes = content[payload_start:payload_start+8]
    data_size = struct.unpack('<Q', size_bytes)[0]
    print(f"Data Payload Size: {data_size} bytes")
    
    # Read float data
    data_bytes = content[payload_start+8 : payload_start+8+data_size]
    data = np.frombuffer(data_bytes, dtype=np.float32)
    
    print(f"Num Values: {len(data)}")
    print(f"Min: {data.min()}")
    print(f"Max: {data.max()}")
    print(f"Mean: {data.mean()}")
    
    # Check bounds
    # Sphere/Cylinder should have negative values inside
    print(f"Count < 0: {np.sum(data < 0)}")
    print(f"Count > 1e5: {np.sum(data > 1e5)} (Background/Uninitialized)")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        for f in sys.argv[1:]:
            inspect_vti(f)
    else:
        inspect_vti("output/sdf/sphere_test.vti")
        inspect_vti("output/sdf/cylinder_test.vti")
        inspect_vti("output/sdf/packing_phi045.vti")
        inspect_vti("output/sdf/cylinder_highres.vti")
        inspect_vti("output/sdf/packing_cylinders_200.vti")
        inspect_vti("output/sdf/periodic_cylinders.vti")
        inspect_vti("output/sdf/dense_packing.vti")
