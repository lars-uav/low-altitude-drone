import time
import torch
import numpy as np
import tensorflow as tf
import timm

# Configurations
input_size = (1, 3, 480, 640)  # Example input shape (batch_size, channels, height, width)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load PyTorch model
def load_pytorch_model(pth_file, device):
    # Recreate the model architecture
    model = timm.create_model("hf_hub:timm/mobilenetv4_conv_small.e2400_r224_in1k", pretrained=False, num_classes=10)
    # Load the state dictionary
    model.load_state_dict(torch.load(pth_file, map_location=device))
    model = model.to(device)
    model.eval()
    return model


# Load TFLite model
def load_tflite_model(tflite_file):
    interpreter = tf.lite.Interpreter(model_path=tflite_file)
    interpreter.allocate_tensors()
    return interpreter

# Inference time for PyTorch model
def pytorch_inference_time(model, input_size, device, num_iterations=100):
    dummy_input = torch.rand(input_size).to(device)
    times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start_time = time.time()
            _ = model(dummy_input)
            times.append(time.time() - start_time)
    avg_time = np.mean(times)
    return avg_time * 1000  # Convert to milliseconds

# Inference time for TFLite model
def tflite_inference_time(interpreter, input_size, num_iterations=100):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    dummy_input = np.random.rand(*input_size).astype(np.float32)
    times = []
    for _ in range(num_iterations):
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        start_time = time.time()
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]['index'])
        times.append(time.time() - start_time)
    avg_time = np.mean(times)
    return avg_time * 1000  # Convert to milliseconds

# Main comparison function
def compare_inference_times(pth_file, tflite_file, input_size, device):
    print("Loading models...")
    pytorch_model = load_pytorch_model(pth_file, device)
    tflite_interpreter = load_tflite_model(tflite_file)

    print("Measuring inference times...")
    pytorch_time = pytorch_inference_time(pytorch_model, input_size, device)
    tflite_time = tflite_inference_time(tflite_interpreter, input_size)

    print("\nInference Time Comparison:")
    print(f"PyTorch Model Inference Time: {pytorch_time:.2f} ms")
    print(f"TFLite Model Inference Time: {tflite_time:.2f} ms")

if __name__ == "__main__":
    # Paths to model files
    pth_file = "/Users/tanmay/Downloads/mobilenetv4_conv_small_model.pth"
    tflite_file = "/Users/tanmay/Downloads/mobilenetv4_conv_small_model.tflite"

    # Compare inference times
    compare_inference_times(pth_file, tflite_file, input_size, device)
