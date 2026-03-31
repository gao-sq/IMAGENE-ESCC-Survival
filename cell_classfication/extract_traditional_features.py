import os
import json
import numpy as np
import cv2
import torch
import openslide
from tqdm import tqdm
from cell_feature_model import get_cell_traditional_feature
import argparse
import sys

# Add CellViT to path
sys.path.append('path/to/CellViT')


def load_cellvit_features(cellvit_path: str):
    """
    Load CellViT features file
    
    Args:
        cellvit_path (str): Path to CellViT .pt file
        
    Returns:
        dict: CellViT data containing x, positions, metadata, etc.
    """
    if not os.path.exists(cellvit_path):
        print(f"Warning: CellViT file not found: {cellvit_path}")
        return None
    
    try:
        data = torch.load(cellvit_path, map_location='cpu')
        
        return data
        
    except Exception as e:
        print(f"Warning: Failed to load CellViT file: {e}")
        return None


def load_cell_image_from_wsi(wsi_path: str, centroid: list, patch_size: int = 256):
    """
    Load cell image from WSI file based on centroid coordinates
    
    Args:
        wsi_path (str): Path to WSI file
        centroid (list): Cell centroid coordinates [x, y]
        patch_size (int): Size of patch to extract around centroid (default: 256)
        
    Returns:
        numpy array: Cell image patch in RGB format
    """
    if not os.path.exists(wsi_path):
        print(f"Warning: WSI file not found: {wsi_path}")
        return None
    
    try:
        # Open WSI file
        wsi = openslide.OpenSlide(wsi_path)
        
        # Convert centroid to integers
        x_center = int(centroid[0])
        y_center = int(centroid[1])
        
        # Calculate patch boundaries
        x_start = max(0, x_center - patch_size // 2)
        y_start = max(0, y_center - patch_size // 2)
        
        # Read patch from WSI
        patch = wsi.read_region((x_start, y_start), 0, (patch_size, patch_size))
        
        # Convert to numpy array (RGB format)
        patch_array = np.array(patch)
        
        return patch_array
        
    except Exception as e:
        print(f"Warning: Failed to load image from WSI: {e}")
        return None


def build_inst_map_from_contour(centroid: list, contour: list, patch_size: int = 256):
    """
    Build instance map from contour with fixed patch size
    
    Args:
        centroid (list): Cell centroid coordinates [x, y]
        contour (list): Contour coordinates [[x1, y1], [x2, y2], ...]
        patch_size (int): Size of the instance map (default: 256)
        
    Returns:
        numpy array: Instance map with fixed patch_size
    """
    if not contour or len(contour) == 0:
        return None

    # Calculate offset to center contour in patch
    offset_x = patch_size // 2 - int(centroid[0])
    offset_y = patch_size // 2 - int(centroid[1])
        
    # Convert contour to numpy array
    contour_array = np.array(contour, dtype=np.int32)

    # Shift contour coordinates to center in patch
    shifted_contour = contour_array + [offset_x, offset_y]

        # Create instance map with fixed patch_size
    inst_map = np.zeros((patch_size, patch_size), dtype=np.uint8)
    
    # Fill contour area with 1
    cv2.fillPoly(inst_map, [shifted_contour], 1)
    
    return inst_map


def extract_traditional_features_from_json(
    cells_json_path: str,
    wsi_path: str = None,
    cells_features_path: str = None,
    patch_size: int = 256,
    output_dir: str = None,
    use_wsi: bool = False,
    merge: bool = False,
):
    """
    Extract traditional features from cells.json using contours
    
    Args:
        cells_json_path (str): Path to cells.json file
        wsi_path (str): Path to WSI file
        cells_features_path (str): Path to CellViT .pt file
        patch_size (int): Size of patch to extract from WSI (default: 256)
        output_path (str): Path to save extracted features (default: cells_with_traditional_features.pt)
        use_wsi (bool): Whether to use WSI features if available
    """
    
    # Load cells.json
    with open(cells_json_path, 'r') as f:
        cells_data = json.load(f)

    # Load CellViT features
    cellvit_data = None
    if cells_features_path and os.path.exists(cells_features_path):
        cellvit_data = load_cellvit_features(cells_features_path)
        print(f"Loaded CellViT data")
    
    # Extract traditional features for each cell
    cells_with_features = []
    
    for cell_info in tqdm(cells_data.get('cells', []), desc="Extracting traditional features"):
        if len(cells_with_features) == 3:
            break
        # Get cell image from WSI
        cell_image = None
        
        if use_wsi and wsi_path is not None and 'centroid' in cell_info:
            cell_image = load_cell_image_from_wsi(
                wsi_path=wsi_path,
                centroid=cell_info['centroid'],
                patch_size=patch_size
            )
        
        # Build inst_map from contour
        inst_map = None
        if 'contour' in cell_info:
            inst_map = build_inst_map_from_contour(cell_info['centroid'], cell_info['contour'], patch_size)
        
        # Extract traditional features
        try:
            # If no cell image is available, generate a dummy one based on inst_map
            if cell_image is None and inst_map is not None:
                # Generate a random grayscale image with same shape as inst_map
                cell_image = np.random.randint(50, 200, inst_map.shape, dtype=np.uint8)
            
            traditional_feature = get_cell_traditional_feature(
                cell_image=cell_image,
                inst_map=inst_map
            )
            
            # Add traditional features to cell info
            cell_info['traditional_feature'] = traditional_feature
            cells_with_features.append(cell_info)
                
        except Exception as e:
            print(f"Error extracting features for cell: {e}")
            cell_info['traditional_feature'] = None
            cells_with_features.append(cell_info)
    
    # Convert cells_with_features to tensors
    features_list = []
    for cell in cells_with_features:
        if 'traditional_feature' in cell and cell['traditional_feature'] is not None:
            features_list.append(cell['traditional_feature'])
        else:
            features_list.append(np.zeros(201))  # Placeholder for missing features
    

    features = torch.tensor(features_list, dtype=torch.float32).squeeze()
    
    if merge:
        cellvit_data.x = torch.cat([cellvit_data.x, features], dim=1)
    else:
        cellvit_data.x = features

    # Save results as torch format
    if output_dir is None:
        output_dir = os.path.join('output', cells_json_path.split('/')[-2], 'cell_detection')

    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, 'cells_with_traditional_features.pt')
    
    # Save as torch file
    torch.save(cellvit_data, output_path)
    
    print(f"Saved {len(cells_with_features)} cells with traditional features to {output_path}")
    
    return output_path


def batch_extract_traditional_features(
    cells_data_root: str,
    wsi_data_root: str = None,
    cells_features_path: str = None,
    patch_size: int = 256,
    output_dir: str = "output",
    use_wsi: bool = False,
    merge: bool = False,
):
    """
    Batch extract traditional features from multiple cells.json files
    
    Args:
        cells_data_root (str): Root directory of cells.json files
        wsi_data_root (str): Root directory of WSI files
        cells_features_path (str): Root directory of CellViT features
        patch_size (int): Size of patch to extract from WSI (default: 256)
        output_dir (str): Output directory to save extracted features (default: "output")
        use_wsi (bool): Whether to use WSI features if available (default: False)

    """
    
    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all cells.json files
    svs_files = []
    for root, dirs, files in os.walk(wsi_data_root):
        for file in files:
            if file.endswith('.svs'):
                svs_files.append(os.path.join(root, file))
    
    print(f"Found {len(svs_files)} .svs files")
    
    # Process each file
    for svs_path in tqdm(svs_files, desc="Processing .svs files"):
        # Infer paths based on filename
        base_name = os.path.basename(svs_path).replace('.svs', '')
        
        # Infer paths
        # cells.json: in preprocessing directory
        cells_json_path = os.path.join(cells_data_root, 'preprocessing', 'cell_detection', 'cells.json')
        # CellViT: in preprocessing directory
        cellvit_path = os.path.join(cells_features_path, 'preprocessing', 'cell_detection', 'cells.pt')
        
        # Extract features
        try:
            result = extract_traditional_features_from_json(
                cells_json_path=cells_json_path,
                wsi_path=svs_path,
                cellvit_path=cellvit_path,
                patch_size=patch_size,
                output_dir=output_dir,
                use_wsi=use_wsi,
                merge=merge
            )
            print(f"✓ Successfully processed {base_name}")
            return True
            
        except Exception as e:
            print(f"✗ Failed to process {base_name}: {str(e)}")
            return False
    
    print(f"Batch extraction completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract traditional features from cells.json')
    
    # Input/output parameters

    # Single file mode
    parser.add_argument('--cells_json', type=str,
                        help='Path to cells.json file')
    parser.add_argument('--wsi_path', type=str,
                        help='Path to WSI file (.svs or .tiff)')
    parser.add_argument('--cells_features_path', type=str,
                        help='Path to cells_features.pt file')
    # Batch mode
    parser.add_argument('--cells_data_root', type=str,
                        help='Root directory containing multiple cells.json files')
    parser.add_argument('--wsi_data_root', type=str,
                        help='WSI images directory')
    parser.add_argument('--cells_features_data_root', type=str,
                        help='cells_features.pt directory')
    parser.add_argument('--use_wsi', action='store_true',
                        default=True,
                        help='Use WSI features if available')
    parser.add_argument('--patch_size', type=int, default=256,
                        help='Size of patch to extract from WSI (default: 256)')
    parser.add_argument('--merge', action='store_true',
                        default=False,
                        help='Merge traditional features with CellViT features')
    parser.add_argument('--output', type=str,
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    # Execute extraction
    if args.cells_json:
        # Single file mode
        extract_traditional_features_from_json(
            cells_json_path=args.cells_json,
            wsi_path=args.wsi_path,
            cells_features_path=args.cells_features_path,
            patch_size=args.patch_size,
            output_dir=args.output,
            use_wsi=args.use_wsi,
            merge=args.merge
        )
    elif args.cells_data_root:
        # Batch mode
        batch_extract_traditional_features(
            cells_data_root=args.cells_data_root,
            wsi_data_root=args.wsi_data_root,
            cells_features_path=args.cells_features_data_path,
            patch_size=args.patch_size,
            output_dir=args.output,
            use_wsi=args.use_wsi,
            merge=args.merge
        )
    else:
        parser.error('Must specify either --cells_json or --cells_data_root')
