// PastureBiomassAnalyzer using GitHub Models API (GPT-4o)
// Replaces Groq API with working GitHub Models API

using System.Text.Json;
using Azure;
using Azure.AI.Inference;
using BioMasaWeb.Services;

namespace BioMasaWeb.Models
{
    public class PastureBiomassAnalyzer
    {
        private readonly ChatCompletionsClient _chatClient;
        private const string MODEL_NAME = "gpt-4o";
        private readonly List<string> _apiKeys = new();
        private readonly IPytorchModelService? _pytorchService;
        
        private const string ANALYSIS_PROMPT = @"
Ești un expert în agronomie și analiza biomasei din pășuni. Analizează imaginea furnizată și estimează componentele biomasei.

SARCINĂ:
Analizează imaginea pășunii și estimează următoarele componente ale biomasei (în grame):

1. **DryGreen** - Masa uscată a părților verzi (furај de calitate)
2. **DryClover** - Masa uscată a trifoiului (dacă este prezent)
3. **DryDead** - Masa materialului mort/uscat (resturi vegetale)
4. **DryTotal** - Biomasa totală uscată
5. **Gdm** - Green Dry Matter (furajul disponibil efectiv)

INSTRUCȚIUNI:
- Analizează densitatea vegetației, culoarea, înălțimea aproximativă
- Identifică prezența trifoiului, ierbii verzi, material mort
- Estimează biomasa pe baza aspectului vizual
- Oferă recomandări pentru managementul pășunii
- Indică nivelul de încredere în estimare (0-1)

RĂSPUNS OBLIGATORIU ÎN FORMAT JSON STRICT:
```json
{
  ""Id"": ""[UUID generat]"",
  ""Title"": ""Analiză Biomasa Pășune"",
  ""Description"": ""[descriere scurtă a stării pășunii]"",
  ""DryGreen"": [valoare numerică],
  ""DryClover"": [valoare numerică],
  ""DryDead"": [valoare numerică],
  ""DryTotal"": [valoare numerică],
  ""Gdm"": [valoare numerică],
  ""Recommendations"": ""[recomandări pentru fermier - 2-3 propoziții]"",
  ""Confidence"": [valoare între 0 și 1]
}
```

[FOARTE IMPORTANT] 
- Răspunde DOAR cu JSON-ul, fără text suplimentar
- Valorile numerice trebuie să fie realiste pentru o pășune
- DryTotal = DryGreen + DryClover + DryDead
- Gdm este aproximativ 85% din DryGreen
- Dacă imaginea NU este o pășune, setează toate valorile la 0 și explică în Recommendations
";

        public PastureBiomassAnalyzer(IPytorchModelService? pytorchService = null)
        {
            _pytorchService = pytorchService;
            
            
            string key = _apiKeys[Random.Shared.Next(0, _apiKeys.Count)];
            
            var endpoint = new Uri("https://models.inference.ai.azure.com");
            var credential = new AzureKeyCredential(key);
            
            _chatClient = new ChatCompletionsClient(
                endpoint,
                credential,
                new AzureAIInferenceClientOptions());
            
            Console.WriteLine("[PastureBiomassAnalyzer] Initialized with GitHub Models API (GPT-4o)");
            Console.WriteLine($"[PastureBiomassAnalyzer] PyTorch integration: {(_pytorchService?.IsModelLoaded == true ? "ENABLED" : "DISABLED")}");
        }

        public async Task<string> AnalyzePastureImageAsync(byte[] imageBytes)
        {
            ArgumentNullException.ThrowIfNull(imageBytes, nameof(imageBytes));

            if (imageBytes.Length == 0)
            {
                throw new BiomassAnalysisException("Imaginea este goală (0 bytes)");
            }

            if (imageBytes.Length < 100)
            {
                throw new BiomassAnalysisException($"Imaginea este prea mică ({imageBytes.Length} bytes) - posibil coruptă");
            }

            try
            {
                Console.WriteLine($"[PastureBiomassAnalyzer] ========== START ANALYSIS ==========");
                Console.WriteLine($"[PastureBiomassAnalyzer] Image bytes length: {imageBytes.Length}");
                Console.WriteLine($"[PastureBiomassAnalyzer] First 10 bytes (hex): {string.Join(" ", imageBytes.Take(10).Select(b => b.ToString("X2")))}");
                
            
                string imageType = DetectImageType(imageBytes);
                Console.WriteLine($"[PastureBiomassAnalyzer] Detected image type: {imageType}");
   
                string pytorchPredictions = "";
                if (_pytorchService?.IsModelLoaded == true)
                {
                    Console.WriteLine("[PastureBiomassAnalyzer] Running PyTorch model inference...");
                    try
                    {
                        var pytorchResult = await _pytorchService.PredictBiomassAsync(imageBytes);
                        if (pytorchResult.Success)
                        {
                            pytorchPredictions = pytorchResult.ToPromptString();
                            Console.WriteLine($"[PastureBiomassAnalyzer] PyTorch predictions obtained successfully");
                            Console.WriteLine($"[PastureBiomassAnalyzer] PyTorch: DryTotal={pytorchResult.DryTotal:F2}g, GDM={pytorchResult.Gdm:F2}g");
                        }
                        else
                        {
                            Console.WriteLine($"[PastureBiomassAnalyzer] PyTorch prediction failed: {pytorchResult.ErrorMessage}");
                        }
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"[PastureBiomassAnalyzer] PyTorch error: {ex.Message}");
             
                    }
                }
                else
                {
                    Console.WriteLine("[PastureBiomassAnalyzer] PyTorch model not available, using Vision AI only");
                }
                
                  string enhancedPrompt = ANALYSIS_PROMPT;
                if (!string.IsNullOrEmpty(pytorchPredictions))
                {
                    enhancedPrompt = pytorchPredictions + "\n\n" + ANALYSIS_PROMPT;
                }
                
               
                var imageBase64 = Convert.ToBase64String(imageBytes);
                var imageDataUrl = $"data:{imageType};base64,{imageBase64}";
                
                var requestOptions = new ChatCompletionsOptions
                {
                    Messages =
                    {
                        new ChatRequestSystemMessage("You are a pasture biomass analysis expert."),
                        new ChatRequestUserMessage(
                            new ChatMessageTextContentItem(enhancedPrompt),
                            new ChatMessageImageContentItem(new Uri(imageDataUrl))
                        )
                    },
                    Model = MODEL_NAME,
                    Temperature = 0.3f,
                    MaxTokens = 1000
                };
                
                Console.WriteLine("[PastureBiomassAnalyzer] Sending request to GitHub Models API...");
                Console.WriteLine($"[PastureBiomassAnalyzer] Request has {requestOptions.Messages.Count} messages");
                
                Response<ChatCompletions> response = await _chatClient.CompleteAsync(requestOptions);
                
                Console.WriteLine($"[PastureBiomassAnalyzer] Response status: {response.GetRawResponse().Status}");
                
                var chatResponse = response.Value;
                if (chatResponse == null)
                {
                    throw new BiomassAnalysisException("Răspuns gol de la GitHub Models API");
                }

          
                var jsonResponse = chatResponse.Content;
                if (string.IsNullOrEmpty(jsonResponse))
                {
          
                    var choices = chatResponse.GetType().GetProperty("Choices")?.GetValue(chatResponse);
                    if (choices != null)
                    {
                        var choicesList = choices as System.Collections.IList;
                        if (choicesList != null && choicesList.Count > 0)
                        {
                            var firstChoice = choicesList[0];
                            var message = firstChoice?.GetType().GetProperty("Message")?.GetValue(firstChoice);
                            jsonResponse = message?.GetType().GetProperty("Content")?.GetValue(message) as string;
                        }
                    }
                }
                
                Console.WriteLine($"[PastureBiomassAnalyzer] JSON Response ({jsonResponse?.Length ?? 0} chars): {jsonResponse?.Substring(0, Math.Min(200, jsonResponse?.Length ?? 0))}...");
                
                return jsonResponse ?? throw new BiomassAnalysisException("Răspuns gol de la GitHub Models API");
            }
            catch (Exception ex) when (ex is not BiomassAnalysisException)
            {
                Console.WriteLine($"[PastureBiomassAnalyzer] ========== ERROR ==========");
                Console.WriteLine($"[PastureBiomassAnalyzer] Exception Type: {ex.GetType().Name}");
                Console.WriteLine($"[PastureBiomassAnalyzer] Exception Message: {ex.Message}");
                Console.WriteLine($"[PastureBiomassAnalyzer] Stack Trace: {ex.StackTrace}");
                throw new BiomassAnalysisException("Eroare la analiza imaginii de pășune cu GitHub Models API", ex);
            }
        }

        private static string DetectImageType(byte[] imageBytes)
        {
     
            if (imageBytes.Length < 4) return "image/jpeg";
            
            // PNG: 89 50 4E 47
            if (imageBytes[0] == 0x89 && imageBytes[1] == 0x50 && imageBytes[2] == 0x4E && imageBytes[3] == 0x47)
                return "image/png";
            
            // JPEG: FF D8 FF
            if (imageBytes[0] == 0xFF && imageBytes[1] == 0xD8 && imageBytes[2] == 0xFF)
                return "image/jpeg";
            
            // GIF: 47 49 46
            if (imageBytes[0] == 0x47 && imageBytes[1] == 0x49 && imageBytes[2] == 0x46)
                return "image/gif";
            
            // Default
            return "image/jpeg";
        }

       
        public BiomassAnalysisResponse ProcessAnalysisResponse(string jsonResponse, string imageBase64)
        {
            try
            {
                Console.WriteLine("[PastureBiomassAnalyzer] Processing JSON response...");
                
                
                var cleanedJson = CleanJsonResponse(jsonResponse);
                
                Console.WriteLine($"[PastureBiomassAnalyzer] Cleaned JSON ({cleanedJson.Length} chars): {cleanedJson.Substring(0, Math.Min(200, cleanedJson.Length))}...");
                
           
                var groqResponse = JsonSerializer.Deserialize<GroqBiomassResponse>(cleanedJson, new JsonSerializerOptions
                {
                    PropertyNameCaseInsensitive = true
                }) ?? throw new JsonException("Deserializare eșuată");

                Console.WriteLine($"[PastureBiomassAnalyzer] Deserialization successful: DryTotal={groqResponse.DryTotal}g, Confidence={groqResponse.Confidence}");

                return new BiomassAnalysisResponse
                {
                    Id = string.IsNullOrEmpty(groqResponse.Id) ? Guid.NewGuid().ToString() : groqResponse.Id,
                    Title = groqResponse.Title,
                    Description = groqResponse.Description,
                    AnalysisDate = DateTime.Now,
                    Components = new BiomassComponents
                    {
                        DryGreenG = groqResponse.DryGreen,
                        DryCloverG = groqResponse.DryClover,
                        DryDeadG = groqResponse.DryDead,
                        DryTotalG = groqResponse.DryTotal,
                        GdmG = groqResponse.Gdm
                    },
                    ImageBase64 = imageBase64,
                    Recommendations = groqResponse.Recommendations,
                    ConfidenceScore = groqResponse.Confidence
                };
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[PastureBiomassAnalyzer] Processing error: {ex.Message}");
                throw new BiomassAnalysisException("Eroare la procesarea răspunsului JSON", ex);
            }
        }

     
        public async Task<BiomassAnalysisResponse> AnalyzeAndProcessAsync(byte[] imageBytes)
        {
            Console.WriteLine("[PastureBiomassAnalyzer] Starting complete analysis...");
            
            var imageBase64 = Convert.ToBase64String(imageBytes);
            var jsonResponse = await AnalyzePastureImageAsync(imageBytes);
            var result = ProcessAnalysisResponse(jsonResponse, imageBase64);
            
            Console.WriteLine("[PastureBiomassAnalyzer] Analysis completed successfully!");
            
            return result;
        }

        private static string CleanJsonResponse(string response)
        {
          
            var cleaned = response.Trim();
            if (cleaned.StartsWith("```json"))
            {
                cleaned = cleaned.Substring(7);
            }
            if (cleaned.StartsWith("```"))
            {
                cleaned = cleaned.Substring(3);
            }
            if (cleaned.EndsWith("```"))
            {
                cleaned = cleaned.Substring(0, cleaned.Length - 3);
            }
            return cleaned.Trim();
        }
    }

   
    public class BiomassAnalysisException : Exception
    {
        public BiomassAnalysisException(string message, Exception innerException)
            : base(message, innerException)
        {
        }

        public BiomassAnalysisException(string message)
            : base(message)
        {
            Console.WriteLine($"[BiomassAnalysisException] {message}");
        }
    }
}

