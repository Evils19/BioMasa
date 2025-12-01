using BioMasaWeb.Models;

namespace BioMasaWeb.Services;

public interface IBiomassAnalysisService
{
    Task<BiomassAnalysisResponse> AnalyzeImageAsync(byte[] imageData);
}

public class BiomassAnalysisService : IBiomassAnalysisService
{
    private readonly PastureBiomassAnalyzer _analyzer;
    private readonly ILogger<BiomassAnalysisService> _logger;

    public BiomassAnalysisService(
        PastureBiomassAnalyzer analyzer,
        ILogger<BiomassAnalysisService> logger)
    {
        _analyzer = analyzer ?? throw new ArgumentNullException(nameof(analyzer));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    public async Task<BiomassAnalysisResponse> AnalyzeImageAsync(byte[] imageData)
    {
        ArgumentNullException.ThrowIfNull(imageData, nameof(imageData));

        const int maxRetries = 2;
        var retryCount = 0;

        while (retryCount <= maxRetries)
        {
            try
            {
                _logger.LogInformation("Începe analiza imaginii ({Size} bytes), încercare {Retry}/{Max}", 
                    imageData.Length, retryCount + 1, maxRetries + 1);
                
                var result = await _analyzer.AnalyzeAndProcessAsync(imageData);
                
                _logger.LogInformation(
                    "Analiză completă: DryTotal={DryTotal}g, GDM={Gdm}g, Confidence={Confidence}", 
                    result.Components.DryTotalG, 
                    result.Components.GdmG,
                    result.ConfidenceScore);
                
                return result;
            }
            catch (TaskCanceledException) when (retryCount < maxRetries)
            {
                retryCount++;
                _logger.LogWarning("Timeout la încercarea {Retry}, reîncerc...", retryCount);
                await Task.Delay(1000 * retryCount); // Exponential backoff
            }
            catch (TaskCanceledException)
            {
                _logger.LogError("Timeout după {Max} încercări", maxRetries + 1);
                throw new TimeoutException($"Groq API nu răspunde după {maxRetries + 1} încercări. Te rog reîncearcă cu o imagine mai mică.");
            }
            catch (HttpRequestException ex)
            {
                _logger.LogError(ex, "Eroare de rețea la apelul Groq API");
                throw new ApplicationException("Eroare de conexiune la serviciul AI. Verifică conexiunea la internet.", ex);
            }
            catch (Exception ex) when (ex.GetType().Name == "GroqApiException")
            {
                _logger.LogError(ex, "Eroare la apelul GitHub Models API");
                throw;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Eroare neașteptată la analiza imaginii");
                throw new ApplicationException("Eroare la procesarea imaginii", ex);
            }
        }

        throw new TimeoutException("Nu s-a putut completa analiza.");
    }
}

