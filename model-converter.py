import ai_edge_torch
import torch
import timm
import numpy as np
from pathlib import Path

def load_pytorch_model(model_path, num_classes=10):
    """
    Load the trained PyTorch model
    """
    model = timm.create_model(
        "hf_hub:timm/mobilenetv4_conv_small.e2400_r224_in1k",
        pretrained=False,
        num_classes=num_classes
    )
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def convert_to_litert(model, input_shape=(1, 3, 480, 640)):
    """
    Convert PyTorch model to LiteRT format
    """
    # Create sample input with the specified shape
    sample_input = (torch.randn(*input_shape),)
    
    # Convert the model using AI Edge Torch
    edge_model = ai_edge_torch.convert(model, sample_input)
    
    return edge_model

def verify_model(edge_model, input_shape=(1, 3, 480, 640)):
    """
    Verify the converted model with a sample input
    """
    # Create test input
    test_input = torch.randn(*input_shape)
    
    # Run inference
    try:
        output = edge_model(test_input)
        print("\nTest inference successful!")
        print(f"Output shape: {output.shape}")
        return True
    except Exception as e:
        print(f"\nError during test inference: {str(e)}")
        return False

def main():
    # Paths
    pytorch_model_path = '/kaggle/input/mobilenetv4/pytorch/default/1/mobilenetv4_conv_small_model.pth'
    output_model_path = 'mobilenetv4_conv_small_model.tflite'
    
    print("Starting conversion process...")
    
    # Load PyTorch model
    print("Loading PyTorch model...")
    pytorch_model = load_pytorch_model(pytorch_model_path)
    
    # Convert to LiteRT
    print("Converting to LiteRT...")
    edge_model = convert_to_litert(pytorch_model)
    
    # Verify the model
    print("Verifying converted model...")
    if verify_model(edge_model):
        # Export the model
        print(f"\nSaving model to {output_model_path}...")
        edge_model.export(output_model_path)
        print("Conversion complete!")
    else:
        print("Model verification failed. Please check the model and conversion process.")

if __name__ == "__main__":
    main()