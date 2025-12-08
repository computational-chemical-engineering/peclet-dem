import base64
import struct
import numpy as np
import sys

filename = "output/particles_0090.vtp"
with open(filename, "r") as f:
    content = f.read()

# Find Radius array
start_tag = '<DataArray type="Float32" Name="Radius" NumberOfComponents="1" format="binary">'
start = content.find(start_tag)
if start == -1:
    print("Radius array not found!")
    sys.exit(1)

print(f"Found Radius array at index {start}")
start_data = content.find('>', start) + 1
end_data = content.find('<', start_data)
b64_data = content[start_data:end_data].strip()

try:
    decoded = base64.b64decode(b64_data)
    # First 4 bytes = header (size)
    header = struct.unpack('i', decoded[:4])[0]
    print(f"Data Header Size: {header} bytes")
    
    data = np.frombuffer(decoded[4:], dtype=np.float32)

    print(f"Count: {len(data)}")
    if len(data) > 0:
        print(f"Min: {data.min()}")
        print(f"Max: {data.max()}")
        print(f"Mean: {data.mean()}")
        print(f"First 10: {data[:10]}")
    else:
        print("Data array is empty!")

except Exception as e:
    print(f"Error decoding: {e}")
