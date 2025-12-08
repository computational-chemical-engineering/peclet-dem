import numpy as np
import base64
import struct

def save_to_vtp(filename, positions, velocities=None, scales=None, types=None):
    """
    Save particle data to a VTP (XML PolyData) file for ParaView.

    Args:
        filename (str): Output filename (e.g., "particles_0000.vtp")
        positions (np.ndarray): (N, 3) float32 array of particle positions.
        velocities (np.ndarray, optional): (N, 3) float32 array of velocities.
        scales (np.ndarray, optional): (N,) float32 array of particle scales/radii.
        types (np.ndarray, optional): (N,) int32/float32 array of particle types.
    """
    num_points = positions.shape[0]

    # Helper to write data array
    def write_data_array(f, name, data, num_components=1):
        dtype = data.dtype
        if dtype == np.float32:
            type_str = "Float32"
        elif dtype == np.int32:
            type_str = "Int32"
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")
        
        f.write(f'        <DataArray type="{type_str}" Name="{name}" NumberOfComponents="{num_components}" format="binary">\n')
        
        # Binary data block: [int32 header size] [data]
        # Header size is the number of bytes in the data block
        raw_data = data.tobytes()
        header = struct.pack('i', len(raw_data))
        encoded_data = base64.b64encode(header + raw_data).decode('utf-8')
        
        f.write(f'          {encoded_data}\n')
        f.write('        </DataArray>\n')

    with open(filename, 'w') as f:
        f.write('<?xml version="1.0"?>\n')
        # Correct header for uncompressed
        f.write('<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">\n')
        f.write(f'  <PolyData>\n')
        f.write(f'    <Piece NumberOfPoints="{num_points}" NumberOfVerts="0" NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="0">\n')
        
        # Points
        f.write('      <Points>\n')
        write_data_array(f, "Points", positions.astype(np.float32), 3)
        f.write('      </Points>\n')
        
        # Point Data
        f.write('      <PointData>\n')
        if velocities is not None:
            write_data_array(f, "Velocity", velocities.astype(np.float32), 3)
        if scales is not None:
            write_data_array(f, "Radius", scales.astype(np.float32), 1)
        if types is not None:
            write_data_array(f, "Type", types.astype(np.int32), 1)
            
        # Add basic ID
        ids = np.arange(num_points, dtype=np.int32)
        write_data_array(f, "ID", ids, 1)
        
        f.write('      </PointData>\n')
        
        # Verts (Particles need to be defined as Verts to show up as points if strictly following PolyData, 
        # but often ParaView shows Points automatically. Let's add Verts explicitly for correctness.)
        # Connectivity for N points: 0, 1, 2, ... N-1
        # Offsets: 1, 2, 3, ... N
        
        # Actually, for just cloud of points, ParaView "Point Gaussian" representation works without cells (Verts).
        # But "Surface" representation needs Verts.
        # Let's avoid adding Verts to keep file size small unless user complaints.
        # Wait, without cells, many filters might ignore them. 
        # But Point Cloud is usually fine.
        
        f.write('    </Piece>\n')
        f.write('  </PolyData>\n')
        f.write('</VTKFile>\n')

if __name__ == "__main__":
    # Simple test
    N = 10
    pos = np.random.rand(N, 3).astype(np.float32)
    vel = np.random.rand(N, 3).astype(np.float32)
    scale = np.random.rand(N).astype(np.float32)
    save_to_vtp("test.vtp", pos, vel, scale)
    print("Saved test.vtp")
