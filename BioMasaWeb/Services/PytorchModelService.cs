using TorchSharp;
using static TorchSharp.torch;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace BioMasaWeb.Services;


public interface IPytorchModelService
{
    Task<PytorchPredictionResult> PredictBiomassAsync(byte[] imageBytes);
    bool IsModelLoaded { get; }
}

public class PytorchModelService : IPytorchModelService, IDisposable
{
    private readonly ILogger<PytorchModelService> _logger;
    private readonly string _modelPath;
    private dynamic? _model;
    private bool _isModelLoaded;
    private readonly Device _device;

    public bool IsModelLoaded => _isModelLoaded;

    public PytorchModelService(ILogger<PytorchModelService> logger, IWebHostEnvironment env)
    {
        _logger = logger;
        _modelPath = Path.Combine(env.ContentRootPath, "Models", "fold0_best.pth");
        
        // Используем CPU (можно изменить на CUDA если доступно)
        _device = torch.cuda.is_available() ? torch.CUDA : torch.CPU;
        _logger.LogInformation($"Using device: {_device.type}");

        try
        {
            LoadModel();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to load PyTorch model on startup. Service will attempt lazy loading.");
            _isModelLoaded = false;
        }
    }

    private void LoadModel()
    {
        try
        {
            _logger.LogInformation($"Loading PyTorch model from: {_modelPath}");

            if (!File.Exists(_modelPath))
            {
                throw new FileNotFoundException($"Model file not found: {_modelPath}");
            }

            // Загружаем модель
            // Примечание: fold0_best.pth может содержать state_dict или полную модель
            try
            {
                _model = torch.jit.load(_modelPath, _device);
            }
            catch
            {
                // Если не работает jit.load, пробуем загрузить как state_dict
                _logger.LogWarning("JIT load failed, attempting to load as state_dict");
                _model = CreateModelArchitecture();
                _model.load(_modelPath);
            }

            _model.eval();
            _isModelLoaded = true;
            
            _logger.LogInformation("PyTorch model loaded successfully");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error loading PyTorch model");
            _isModelLoaded = false;
            throw;
        }
    }

    private nn.Module<Tensor, Tensor> CreateModelArchitecture()
    {
        // Создаем простую ResNet-like архитектуру
        // Это упрощенная версия - может потребоваться корректировка
        _logger.LogInformation("Creating model architecture");

        var model = nn.Sequential(
            ("conv1", nn.Conv2d(3, 64, kernelSize: 7, stride: 2, padding: 3)),
            ("relu1", nn.ReLU()),
            ("pool1", nn.MaxPool2d(kernelSize: 3, stride: 2, padding: 1)),
            ("conv2", nn.Conv2d(64, 128, kernelSize: 3, stride: 1, padding: 1)),
            ("relu2", nn.ReLU()),
            ("pool2", nn.MaxPool2d(kernelSize: 2, stride: 2)),
            ("flatten", nn.Flatten()),
            ("fc1", nn.Linear(128 * 56 * 56, 512)),
            ("relu3", nn.ReLU()),
            ("fc2", nn.Linear(512, 5)) // 5 выходов: DryGreen, DryClover, DryDead, DryTotal, GDM
        );

        return model;
    }

    public async Task<PytorchPredictionResult> PredictBiomassAsync(byte[] imageBytes)
    {
        if (!_isModelLoaded || _model == null)
        {
            _logger.LogWarning("Model not loaded, attempting to load now");
            LoadModel();
            
            if (!_isModelLoaded || _model == null)
            {
                throw new InvalidOperationException("PyTorch model is not loaded and could not be loaded");
            }
        }

        try
        {
            _logger.LogInformation($"Processing image with PyTorch model ({imageBytes.Length} bytes)");

       
            var imageTensor = await PreprocessImageAsync(imageBytes);

            using (torch.no_grad())
            {
                var output = _model.forward(imageTensor);
                var predictions = output.cpu().data<float>().ToArray();

                var predictionStrings = new string[predictions.Length];
                for (int i = 0; i < predictions.Length; i++)
                {
                    predictionStrings[i] = predictions[i].ToString("F2");
                }
                _logger.LogInformation($"PyTorch predictions: [{string.Join(", ", predictionStrings)}]");

                // Создаем результат
                var result = new PytorchPredictionResult
                {
                    DryGreen = Math.Max(0, predictions.Length > 0 ? predictions[0] : 0),
                    DryClover = Math.Max(0, predictions.Length > 1 ? predictions[1] : 0),
                    DryDead = Math.Max(0, predictions.Length > 2 ? predictions[2] : 0),
                    DryTotal = Math.Max(0, predictions.Length > 3 ? predictions[3] : 0),
                    Gdm = Math.Max(0, predictions.Length > 4 ? predictions[4] : 0),
                    RawPredictions = predictions.ToList(),
                    Success = true
                };

                // Вычисляем confidence (простая метрика)
                result.Confidence = CalculateConfidence(predictions);

                return result;
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error during PyTorch model prediction");
            
            return new PytorchPredictionResult
            {
                Success = false,
                ErrorMessage = ex.Message
            };
        }
    }

    private async Task<Tensor> PreprocessImageAsync(byte[] imageBytes)
    {
        return await Task.Run(() =>
        {
            // Загружаем изображение с помощью ImageSharp
            using var image = Image.Load<Rgb24>(imageBytes);
            
            // Изменяем размер до 224x224 (стандартный размер для ResNet)
            image.Mutate(x => x.Resize(224, 224));

            // Конвертируем в тензор [1, 3, 224, 224]
            var tensor = torch.zeros(new long[] { 1, 3, 224, 224 }, dtype: ScalarType.Float32);
            
            var tensorSpan = tensor.data<float>();
            int idx = 0;

            // Нормализация ImageNet: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            float[] mean = { 0.485f, 0.456f, 0.406f };
            float[] std = { 0.229f, 0.224f, 0.225f };

            for (int c = 0; c < 3; c++) // Каналы R, G, B
            {
                for (int y = 0; y < 224; y++)
                {
                    for (int x = 0; x < 224; x++)
                    {
                        var pixel = image[x, y];
                        float value = c switch
                        {
                            0 => pixel.R / 255f,
                            1 => pixel.G / 255f,
                            2 => pixel.B / 255f,
                            _ => 0f
                        };
                        
                        // Применяем нормализацию
                        tensorSpan[idx++] = (value - mean[c]) / std[c];
                    }
                }
            }

            return tensor.to(_device);
        });
    }

    private double CalculateConfidence(float[] predictions)
    {
        // Простая метрика уверенности
        // Можно улучшить в зависимости от специфики модели
        if (predictions.Length == 0) return 0.5;

        // Используем вариацию predictions как индикатор уверенности
        var mean = predictions.Average();
        var variance = predictions.Select(p => Math.Pow(p - mean, 2)).Average();
        var confidence = 1.0 / (1.0 + variance);

        return Math.Clamp(confidence, 0.1, 0.95);
    }

    public void Dispose()
    {
        _model?.Dispose();
        _logger.LogInformation("PytorchModelService disposed");
    }
}


public class PytorchPredictionResult
{
    public double DryGreen { get; set; }
    public double DryClover { get; set; }
    public double DryDead { get; set; }
    public double DryTotal { get; set; }
    public double Gdm { get; set; }
    public double Confidence { get; set; }
    public List<float> RawPredictions { get; set; } = new();
    public bool Success { get; set; }
    public string? ErrorMessage { get; set; }
    
   
    public string ToPromptString()
    {
        return $@"
PyTorch Model Analysis Results:
- Dry Green Biomass (predicted): {DryGreen:F2}g
- Dry Clover Biomass (predicted): {DryClover:F2}g
- Dry Dead Material (predicted): {DryDead:F2}g
- Total Dry Biomass (predicted): {DryTotal:F2}g
- GDM - Green Dry Matter (predicted): {Gdm:F2}g
- Model Confidence: {Confidence:P0}

Please analyze this pasture image and refine these predictions based on visual analysis.
Provide realistic estimates and recommendations in Romanian language.
";
    }
}

/// <summary>
/// Null Object Pattern - используется когда PyTorch модель не загружается
/// </summary>
public class NullPytorchModelService : IPytorchModelService
{
    public bool IsModelLoaded => false;

    public Task<PytorchPredictionResult> PredictBiomassAsync(byte[] imageBytes)
    {
        return Task.FromResult(new PytorchPredictionResult
        {
            Success = false,
            ErrorMessage = "PyTorch model not loaded"
        });
    }
}
