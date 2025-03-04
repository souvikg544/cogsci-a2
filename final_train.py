import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torchvision.transforms as T
from PIL import Image
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold
import random
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import pickle
from sklearn.base import clone

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


os.environ['TORCH_HOME'] = '/ssd_scratch/cvit/souvik/torch_cache'
os.makedirs('/ssd_scratch/cvit/souvik/torch_cache', exist_ok=True)

# Check for GPU availability and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# If multiple GPUs are available, let's use them
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")

# Configuration
SUBJECT_ID = 4
ROIS = ['FBA-2', 'VWFA-1']  # V3d from early retinotopic visual regions, OPA from place-selective regions
VERTICES_PER_ROI = 10
TEST_SIZE = 0.2
EPOCHS = 50             # Fixed number of epochs (changed from MAX_EPOCHS)
BATCH_SIZE = 32         # Larger batch size for GPU processing

# Paths (Updated based on your local setup)
BASE_DIR = '/ssd_scratch/cvit/souvik/'
TRAIN_DIR = os.path.join(BASE_DIR, f'subj04/training_split')
STIMULI_DIR = os.path.join(TRAIN_DIR, 'training_images')
FMRI_DIR = os.path.join(TRAIN_DIR, 'training_fmri')
ROI_MASKS_DIR = os.path.join(BASE_DIR, f'subj04/roi_masks/')

# Part 1: Preparing the Dataset

class BrainEncodingDataset(Dataset):
    def __init__(self, image_paths, brain_responses=None, transform=None):
        self.image_paths = image_paths
        self.brain_responses = brain_responses
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        if self.brain_responses is not None:
            return image, self.brain_responses[idx]
        else:
            return image

def load_roi_mapping():
    """Create a mapping of ROIs to their category directory."""
    roi_mapping = {
        # Early retinotopic visual regions
        'V1v': 'prf-visualrois', 'V1d': 'prf-visualrois', 
        'V2v': 'prf-visualrois', 'V2d': 'prf-visualrois',
        'V3v': 'prf-visualrois', 'V3d': 'prf-visualrois', 
        'hV4': 'prf-visualrois',
        
        # Body-selective regions
        'EBA': 'floc-bodies', 'FBA-1': 'floc-bodies', 
        'FBA-2': 'floc-bodies', 'mTL-bodies': 'floc-bodies',
        
        # Face-selective regions
        'OFA': 'floc-faces', 'FFA-1': 'floc-faces', 
        'FFA-2': 'floc-faces', 'mTL-faces': 'floc-faces', 
        'aTL-faces': 'floc-faces',
        
        # Place-selective regions
        'OPA': 'floc-places', 'PPA': 'floc-places', 'RSC': 'floc-places',
        
        # Word-selective regions
        'OWFA': 'floc-words', 'VWFA-1': 'floc-words', 
        'VWFA-2': 'floc-words', 'mfs-words': 'floc-words', 
        'mTL-words': 'floc-words'
    }
    return roi_mapping

def load_brain_data():
    """Load full brain data for left and right hemispheres."""
    lh_fmri_path = os.path.join(FMRI_DIR, 'lh_training_fmri.npy')
    rh_fmri_path = os.path.join(FMRI_DIR, 'rh_training_fmri.npy')
    
    lh_fmri = np.load(lh_fmri_path)
    rh_fmri = np.load(rh_fmri_path)
    
    print(f"Loaded left hemisphere fMRI data: {lh_fmri.shape}")
    print(f"Loaded right hemisphere fMRI data: {rh_fmri.shape}")
    
    return lh_fmri, rh_fmri

def load_roi_masks():
    """Load ROI masks for subject 3."""
    roi_mapping = load_roi_mapping()
    roi_data = {}
    
    for roi in ROIS:
        roi_dir = roi_mapping[roi]
        
        # Load masks for left and right hemispheres
        lh_mask_path = os.path.join(ROI_MASKS_DIR, f'lh.{roi_dir}_challenge_space.npy')
        rh_mask_path = os.path.join(ROI_MASKS_DIR, f'rh.{roi_dir}_challenge_space.npy')
        
        lh_mask = np.load(lh_mask_path)
        rh_mask = np.load(rh_mask_path)
        
        print(f"Loaded {roi} masks - LH: {lh_mask.shape}, RH: {rh_mask.shape}")
        
        roi_data[roi] = {
            'lh_mask': lh_mask,
            'rh_mask': rh_mask
        }
    
    return roi_data

def extract_vertex_indices(roi, roi_dir, mapping_dir):
    """Extract vertex indices for a specific ROI using mapping files."""
    mapping_file = os.path.join(mapping_dir, f'mapping_{roi_dir}.npy')
    
    if not os.path.exists(mapping_file):
        raise FileNotFoundError(f"Mapping file not found: {mapping_file}")
    
    mapping_data = np.load(mapping_file, allow_pickle=True).item()
    
    if roi not in mapping_data:
        raise ValueError(f"ROI {roi} not found in mapping data")
    
    return mapping_data[roi]['lh'], mapping_data[roi]['rh']

def extract_roi_data(lh_fmri, rh_fmri, roi_masks):
    """Extract fMRI data for specific ROIs using masks and vertex indices."""
    roi_mapping = load_roi_mapping()
    roi_fmri_data = {}
    
    for roi, masks in roi_masks.items():
        # Try to load vertex indices from mapping files
        try:
            lh_indices, rh_indices = extract_vertex_indices(roi, roi_mapping[roi], ROI_MASKS_DIR)
            print(f"Loaded ROI {roi} vertex indices from mapping file")
        except (FileNotFoundError, ValueError):
            # Fall back to using masks directly
            print(f"Using mask-based extraction for ROI {roi}")
            lh_mask = masks['lh_mask']
            rh_mask = masks['rh_mask']
            
            # Get the indices where the mask is True (vertices belonging to the ROI)
            lh_indices = np.where(lh_mask)[0]
            rh_indices = np.where(rh_mask)[0]
        
        # Extract the fMRI data for those vertices
        lh_roi_data = lh_fmri[:, lh_indices]
        rh_roi_data = rh_fmri[:, rh_indices]
        
        # Combine left and right hemisphere data
        combined_roi_data = np.concatenate([lh_roi_data, rh_roi_data], axis=1)
        
        # Select random vertices if there are enough
        num_vertices = combined_roi_data.shape[1]
        if num_vertices < VERTICES_PER_ROI:
            print(f"Warning: ROI {roi} has fewer than {VERTICES_PER_ROI} vertices ({num_vertices})")
            # Use all available vertices
            selected_roi_data = combined_roi_data
            random_indices = np.arange(num_vertices)
        else:
            random_indices = np.random.choice(num_vertices, VERTICES_PER_ROI, replace=False)
            selected_roi_data = combined_roi_data[:, random_indices]
        
        roi_fmri_data[roi] = {
            'data': selected_roi_data,
            'indices': random_indices,
            'full_data': combined_roi_data,
            'lh_indices': lh_indices,
            'rh_indices': rh_indices
        }
        
        print(f"Extracted {roi} data: {combined_roi_data.shape}, selected {len(random_indices)} vertices")
    
    return roi_fmri_data

def load_data():
    """Load image paths and brain data for ROIs."""
    print("Loading data...")
    
    # Load image paths
    train_img_paths = [os.path.join(STIMULI_DIR, img) for img in sorted(os.listdir(STIMULI_DIR)) if img.endswith(('.jpg', '.png'))]
    print(f"Found {len(train_img_paths)} training images")
    
    # Load fMRI data
    lh_fmri, rh_fmri = load_brain_data()
    
    # Load ROI masks
    roi_masks = load_roi_masks()
    
    # Extract ROI-specific data
    roi_fmri_data = extract_roi_data(lh_fmri, rh_fmri, roi_masks)
    
    return train_img_paths, roi_fmri_data

# Part 2: Feature Extraction using Different Architectures

def get_models():
    """Initialize and move CNN models to GPU if available."""
    print("Initializing CNN models...")

    # ResNet-50
    resnet = models.resnet50(weights='IMAGENET1K_V2')
    # EfficientNet
    efficientnet = models.efficientnet_b0(weights='IMAGENET1K_V1')
    # SqueezeNet
    squeezenet = models.squeezenet1_1(weights='IMAGENET1K_V1')
    
    # Use DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        resnet = torch.nn.DataParallel(resnet)
        efficientnet = torch.nn.DataParallel(efficientnet)
        squeezenet = torch.nn.DataParallel(squeezenet)
    
    # Move models to device
    resnet = resnet.to(device)
    efficientnet = efficientnet.to(device)
    squeezenet = squeezenet.to(device)
    
    # Set to eval mode
    resnet.eval()
    efficientnet.eval()
    squeezenet.eval()

    return {
        'ResNet-50': resnet,
        'EfficientNet': efficientnet,
        'SqueezeNet': squeezenet
    }


def extract_features(model_name, model, image_paths):
    """Extract features from the final layer of the model."""
    
    print(f"Extracting features using {model_name}...")
    
    # Define the transformation appropriate for each model
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Create dataset and dataloader with more workers for parallel processing
    dataset = BrainEncodingDataset(image_paths, transform=preprocess)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)
    
    features = []
    
    with torch.no_grad():
        for images in tqdm(dataloader):
            # Move images to device
            images = images.to(device)
            
            if model_name == 'ResNet-50':
                # Account for DataParallel wrapper
                if isinstance(model, torch.nn.DataParallel):
                    module = model.module
                else:
                    module = model
                    
                # Remove the final classification layer
                x = module.conv1(images)
                x = module.bn1(x)
                x = module.relu(x)
                x = module.maxpool(x)
                x = module.layer1(x)
                x = module.layer2(x)
                x = module.layer3(x)
                x = module.layer4(x)
                x = module.avgpool(x)
                batch_features = torch.flatten(x, 1)
            
            elif model_name == 'EfficientNet':
                # Extract features using forward pass without classification head
                if isinstance(model, torch.nn.DataParallel):
                    module = model.module
                else:
                    module = model
                    
                batch_features = module.features(images)
                batch_features = module.avgpool(batch_features)
                batch_features = torch.flatten(batch_features, 1)
            
            elif model_name == 'SqueezeNet':
                # Extract features before the classifier
                if isinstance(model, torch.nn.DataParallel):
                    module = model.module
                else:
                    module = model
                    
                batch_features = module.features(images)
                batch_features = torch.nn.functional.adaptive_avg_pool2d(batch_features, (1, 1))
                batch_features = torch.flatten(batch_features, 1)
            
            features.append(batch_features.cpu().numpy())
    
    features_array = np.vstack(features)
    print(f"Extracted {features_array.shape[0]} feature vectors with {features_array.shape[1]} dimensions")
    
    # Save features to a file
    os.makedirs('features', exist_ok=True)
    with open(f'features/{model_name}_features.pkl', 'wb') as f:
        pickle.dump(features_array, f)
    
    return features_array

# Part 3: Training Ridge Regression Models with Fixed Epochs (No Early Stopping)

class FixedEpochsRidge:
    """Ridge regression with a fixed number of training iterations."""
    
    def __init__(self, epochs=EPOCHS, alpha=1.0):
        self.epochs = epochs
        self.alpha = alpha
        self.final_model = None
        
    def fit(self, X_train, y_train):
        """Fit model for a fixed number of epochs."""
        # Initialize the model with the specified alpha
        model = Ridge(alpha=self.alpha, fit_intercept=True)
        
        # Train for specified number of epochs (in this case, we just fit once
        # since Ridge regression normally converges immediately)
        for epoch in range(self.epochs):
            # For Ridge regression, we can adjust regularization across epochs if needed
            current_alpha = self.alpha * (1 - epoch/self.epochs)
            model = Ridge(alpha=current_alpha, fit_intercept=True)
            model.fit(X_train, y_train)
            
            # In a real iterative scenario, we might update weights or do other operations
            # but Ridge regression doesn't typically benefit from this kind of iteration
        
        self.final_model = model
        return self.final_model

def train_encoding_models(features_dict, roi_data):
    """Train encoding models for each ROI and architecture with a fixed number of epochs."""
    
    results = {}
    
    for roi, roi_info in roi_data.items():
        roi_results = {}
        brain_responses = roi_info['data']
        
        print(f"\nTraining encoding models for ROI: {roi}")
        
        for model_name, features in features_dict.items():
            vertex_results = []
            
            # For each vertex in the ROI
            for vertex_idx in range(brain_responses.shape[1]):
                print(f"Training model for {roi}, {model_name}, vertex {vertex_idx+1}/{brain_responses.shape[1]}")
                
                # Get the responses for this vertex
                vertex_responses = brain_responses[:, vertex_idx]
                
                # Use k-fold cross-validation
                kf = KFold(n_splits=5, shuffle=True, random_state=42)
                fold_correlations = []
                fold_r2s = []
                
                for fold, (train_idx, test_idx) in enumerate(kf.split(features)):
                    X_train_fold, X_test_fold = features[train_idx], features[test_idx]
                    y_train_fold, y_test_fold = vertex_responses[train_idx], vertex_responses[test_idx]
                    
                    # Train with fixed epochs
                    model = FixedEpochsRidge()
                    final_model = model.fit(X_train_fold, y_train_fold)
                    
                    # Make predictions
                    y_pred = final_model.predict(X_test_fold)
                    
                    # Calculate metrics
                    correlation = np.corrcoef(y_test_fold, y_pred)[0, 1]
                    r2 = r2_score(y_test_fold, y_pred)
                    
                    fold_correlations.append(correlation)
                    fold_r2s.append(r2)
                
                # Average the results across folds
                avg_correlation = np.mean(fold_correlations)
                avg_r2 = np.mean(fold_r2s)
                
                vertex_results.append({
                    'vertex_index': vertex_idx if vertex_idx < len(roi_info['indices']) else roi_info['indices'][0],
                    'correlation': avg_correlation,
                    'r2': avg_r2,
                    'fold_correlations': fold_correlations,
                    'fold_r2s': fold_r2s
                })
            
            # Average the results across vertices
            avg_correlation = np.mean([v['correlation'] for v in vertex_results])
            avg_r2 = np.mean([v['r2'] for v in vertex_results])
            
            roi_results[model_name] = {
                'vertex_results': vertex_results,
                'avg_correlation': avg_correlation,
                'avg_r2': avg_r2
            }
            
            print(f"{model_name} - Avg Correlation: {avg_correlation:.4f}, Avg R²: {avg_r2:.4f}")
        
        results[roi] = roi_results
    
    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/encoding_results_fixed_epochs.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    return results

# Part 4: Analysis and Visualization

def analyze_results(results):
    """Analyze and visualize the results."""
    
    # Prepare data for the bar chart
    roi_names = []
    model_names = []
    correlations = []
    r2_scores = []
    
    for roi, roi_results in results.items():
        for model_name, model_results in roi_results.items():
            roi_names.append(roi)
            model_names.append(model_name)
            correlations.append(model_results['avg_correlation'])
            r2_scores.append(model_results['avg_r2'])
    
    # Create a DataFrame for easier plotting
    df = pd.DataFrame({
        'ROI': roi_names,
        'Model': model_names,
        'Correlation': correlations,
        'R²': r2_scores
    })
    
    # Create output directory for figures
    os.makedirs('figures', exist_ok=True)
    
    # Plot correlation and R² results
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    sns.barplot(x='ROI', y='Correlation', hue='Model', data=df)
    plt.title('Average Correlation by ROI and Model')
    plt.ylim(0, max(correlations) * 1.2)
    
    plt.subplot(1, 2, 2)
    sns.barplot(x='ROI', y='R²', hue='Model', data=df)
    plt.title('Average R² by ROI and Model')
    plt.ylim(0, max(r2_scores) * 1.2)
    
    plt.tight_layout()
    plt.savefig('figures/roi_model_comparison.png')
    plt.show()
    
    # Detailed analysis of vertex performance
    for roi, roi_results in results.items():
        plt.figure(figsize=(15, 5))
        
        vertex_dfs = []
        
        for model_name, model_results in roi_results.items():
            vertex_data = model_results['vertex_results']
            vertex_df = pd.DataFrame(vertex_data)
            vertex_df['Model'] = model_name
            vertex_dfs.append(vertex_df)
        
        vertex_df = pd.concat(vertex_dfs)
        
        plt.subplot(1, 2, 1)
        sns.boxplot(x='Model', y='correlation', data=vertex_df)
        plt.title(f'{roi} - Correlation Distribution by Model')
        
        plt.subplot(1, 2, 2)
        sns.boxplot(x='Model', y='r2', data=vertex_df)
        plt.title(f'{roi} - R² Distribution by Model')
        
        plt.tight_layout()
        plt.savefig(f'figures/{roi}_vertex_distribution.png')
        plt.show()
    
    # Create a heatmap of correlations per vertex
    for roi, roi_results in results.items():
        num_vertices = len(roi_results[list(roi_results.keys())[0]]['vertex_results'])
        plt.figure(figsize=(12, 8))
        
        # Extract correlation data for each model and vertex
        vertex_data = np.zeros((num_vertices, len(roi_results)))
        model_names_list = list(roi_results.keys())
        
        for model_idx, model_name in enumerate(model_names_list):
            model_results = roi_results[model_name]
            for vertex_idx, vertex_result in enumerate(model_results['vertex_results']):
                vertex_data[vertex_idx, model_idx] = vertex_result['correlation']
        
        # Plot heatmap
        sns.heatmap(vertex_data, annot=True, fmt=".3f", cmap="viridis", 
                    xticklabels=model_names_list, 
                    yticklabels=[f"Vertex {i+1}" for i in range(num_vertices)])
        plt.title(f'{roi} - Correlation by Vertex and Model')
        plt.tight_layout()
        plt.savefig(f'figures/{roi}_correlation_heatmap.png')
        plt.show()
    
    # Find the best performing model for each ROI
    best_models = {}
    for roi, roi_results in results.items():
        best_model = max(roi_results.items(), key=lambda x: x[1]['avg_correlation'])
        best_models[roi] = {
            'model': best_model[0],
            'correlation': best_model[1]['avg_correlation'],
            'r2': best_model[1]['avg_r2']
        }
    
    print("\nBest performing models by ROI:")
    for roi, info in best_models.items():
        print(f"{roi}: {info['model']} (Correlation: {info['correlation']:.4f}, R²: {info['r2']:.4f})")
    
    # Additional analysis: Cross-model performance comparison
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Correlation', y='R²', hue='Model', size='ROI', data=df, sizes={'FBA-2': 100, 'VWFA-1': 200})
    plt.title('Model Performance: Correlation vs R²')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('figures/correlation_vs_r2.png')
    plt.show()
    
    # Save results to a CSV for easy reference
    df.to_csv('results/model_comparison_results.csv', index=False)
    
    return df, best_models

# Main execution

def main():
    print("Starting Brain Encoding Models Comparison with Fixed Training Epochs")
    
    # Step 1: Load data
    image_paths, roi_data = load_data()
    
    # Step 2: Extract features from different architectures
    models = get_models()
    
    features_dict = {}
    
    # Check if features are already extracted
    if os.path.exists('features'):
        print("Checking for pre-extracted features...")
        for model_name in models.keys():
            feature_path = f'features/{model_name}_features.pkl'
            if os.path.exists(feature_path):
                print(f"Loading pre-extracted features for {model_name}...")
                with open(feature_path, 'rb') as f:
                    features_dict[model_name] = pickle.load(f)
            else:
                features = extract_features(model_name, models[model_name], image_paths)
                features_dict[model_name] = features
    else:
        # Extract features for all models
        for model_name, model in models.items():
            features = extract_features(model_name, model, image_paths)
            features_dict[model_name] = features
    
    # Step 3: Train encoding models with fixed number of epochs
    if os.path.exists('results/encoding_results_fixed_epochs.pkl'):
        print("Loading previously computed results with fixed epochs...")
        with open('results/encoding_results_fixed_epochs.pkl', 'rb') as f:
            results = pickle.load(f)
    else:
        results = train_encoding_models(features_dict, roi_data)
    
    # Step 4: Analyze and visualize results
    df, best_models = analyze_results(results)
    
    print("\nEncoding model comparison completed!")
    print(f"Used device: {device}")
    if torch.cuda.is_available():
        print(f"GPU(s) used: {torch.cuda.get_device_name(0)}")
    
    return results, df, best_models

if __name__ == "__main__":
    results, df, best_models = main()