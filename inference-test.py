import time
import torch
import numpy as np
import tensorflow as tf
import timm
from pathlib import Path
import torchvision.models.resnet

# Add ResNet to safe globals for weights loading
torch.serialization.add_safe_globals([torchvision.models.resnet.ResNet])

class ModelBenchmark:
    def __init__(self, input_size=(1, 3, 480, 640)):
        self.input_size = input_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    def load_pytorch_model(self, model_name, weights_path, num_classes=10):
        """
        Load a PyTorch model with pre-trained weights.
        """
        model_mapping = {
            "mobilenet_v2": "mobilenetv2_100",
            "mobilenet_v3": "mobilenetv3_small_100",
            "resnet34": "resnet34",
            "mobilenet_v4": "hf_hub:timm/mobilenetv4_conv_small.e2400_r224_in1k"
        }

        if model_name not in model_mapping:
            raise ValueError(f"Unsupported model name: {model_name}")

        # Create model
        model = timm.create_model(
            model_mapping[model_name],
            pretrained=False,
            num_classes=num_classes
        )

        # Load weights if path is provided
        if weights_path:
            weights_path = Path(weights_path)
            if not weights_path.exists():
                raise FileNotFoundError(f"Weights file not found: {weights_path}")
            
            try:
                # First try loading with weights_only=True
                try:
                    checkpoint = torch.load(weights_path, map_location=self.device, weights_only=True)
                except Exception as e:
                    print("Falling back to regular load method due to:", str(e))
                    checkpoint = torch.load(weights_path, map_location=self.device)
                
                # Handle different saving formats
                if isinstance(checkpoint, dict):
                    if 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    elif 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    else:
                        state_dict = checkpoint
                else:
                    state_dict = checkpoint

                # Remove module prefix if present
                new_state_dict = {}
                for k, v in state_dict.items():
                    name = k.replace('module.', '') if k.startswith('module.') else k
                    new_state_dict[name] = v

                # Load the processed state dict
                model.load_state_dict(new_state_dict, strict=False)
                print(f"Successfully loaded weights from {weights_path}")
                
            except Exception as e:
                print(f"Detailed error while loading weights: {str(e)}")
                print("Attempting to continue with default initialization...")

        model = model.to(self.device)
        model.eval()
        return model

    def load_tflite_model(self, tflite_path):
        """
        Load a TFLite model with error handling.
        """
        tflite_path = Path(tflite_path)
        if not tflite_path.exists():
            raise FileNotFoundError(f"TFLite model not found: {tflite_path}")

        try:
            interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
            interpreter.allocate_tensors()
            print(f"Successfully loaded TFLite model from {tflite_path}")
            return interpreter
        except Exception as e:
            raise Exception(f"Error loading TFLite model: {str(e)}")

    def pytorch_inference_time(self, model, warmup=10, num_iterations=100):
        """
        Measure PyTorch model inference time with warmup.
        """
        # Ensure input dimensions match model expectations
        if model.__class__.__name__ == 'ResNet':
            dummy_input = torch.rand((1, 3, 224, 224)).to(self.device)
        else:
            dummy_input = torch.rand(self.input_size).to(self.device)
        
        times = []

        # Warmup runs
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(dummy_input)

        # Actual timing
        with torch.no_grad():
            for _ in range(num_iterations):
                start_time = time.perf_counter()
                _ = model(dummy_input)
                torch.cuda.synchronize() if self.device.type == 'cuda' else None
                times.append(time.perf_counter() - start_time)

        return np.mean(times) * 1000  # Convert to milliseconds

    def tflite_inference_time(self, interpreter, warmup=10, num_iterations=100):
        """
        Measure TFLite model inference time with warmup.
        """
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Get input shape from model
        tflite_input_shape = input_details[0]['shape']
        dummy_input = np.random.rand(*tflite_input_shape).astype(np.float32)
        
        # Warmup runs
        for _ in range(warmup):
            interpreter.set_tensor(input_details[0]['index'], dummy_input)
            interpreter.invoke()

        times = []
        for _ in range(num_iterations):
            interpreter.set_tensor(input_details[0]['index'], dummy_input)
            start_time = time.perf_counter()
            interpreter.invoke()
            _ = interpreter.get_tensor(output_details[0]['index'])
            times.append(time.perf_counter() - start_time)

        return np.mean(times) * 1000  # Convert to milliseconds

    def run_benchmark(self, model_configs):
        """
        Run benchmarks for multiple models.
        """
        results = []
        
        for config in model_configs:
            try:
                print(f"\nBenchmarking {config['name']}...")
                
                # Load PyTorch model
                pytorch_model = self.load_pytorch_model(
                    config['name'],
                    config['weights_path'],
                    config.get('num_classes', 10)
                )
                
                # Load TFLite model
                tflite_interpreter = self.load_tflite_model(config['tflite_path'])
                
                # Run benchmarks
                pytorch_time = self.pytorch_inference_time(pytorch_model)
                tflite_time = self.tflite_inference_time(tflite_interpreter)
                
                results.append({
                    "model_name": config['name'],
                    "pytorch_time_ms": pytorch_time,
                    "tflite_time_ms": tflite_time
                })
                
            except Exception as e:
                print(f"Error benchmarking {config['name']}: {str(e)}")
                continue
        
        return results
        
if __name__ == "__main__":
    # Example configuration
    model_configs = [
        {
            "name": "mobilenet_v2",
            "weights_path": "/kaggle/input/mobilenet-model/pytorch/default/1/mobilenet_model.pth",
            "tflite_path": "/kaggle/input/mobilenet-v2-tflite/tflite/default/1/mobilenet_v2.tflite",
            "num_classes": 10
        },
        {
            "name": "mobilenet_v4",
            "weights_path": "/kaggle/input/mobilenetv4/pytorch/default/1/mobilenetv4_conv_small_model.pth",
            "tflite_path": "/kaggle/input/mobilenet-v4-best-tflite/tflite/default/1/mobilenetv4_conv_small_model_epoch_44.tflite",
            "num_classes": 10
        },
        {
            "name": "resnet34",
            "weights_path": "/kaggle/input/resnet34-rice/pytorch/default/1/resnet34_full_rice_weights.pth",
            "tflite_path": "/kaggle/input/resnet-34-rice-tflite/tflite/default/1/resnet34_rice.tflite",
            "num_classes": 10
        },
        
    ]

    # Initialize and run benchmark
    benchmark = ModelBenchmark()
    results = benchmark.run_benchmark(model_configs)

    # Print results
    print("\nBenchmark Results:")
    print("-" * 50)
    for result in results:
        print(f"\nModel: {result['model_name']}")
        print(f"PyTorch Inference Time: {result['pytorch_time_ms']:.2f} ms")
        print(f"TFLite Inference Time: {result['tflite_time_ms']:.2f} ms")