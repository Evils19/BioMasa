using BioMasaWeb.Components;
using BioMasaWeb.Services;
using BioMasaWeb.Models;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
builder.Services.AddRazorComponents()
    .AddInteractiveServerComponents();

// Register PyTorch Model Service (with graceful fallback if model fails to load)
builder.Services.AddSingleton<IPytorchModelService>(sp =>
{
    var logger = sp.GetRequiredService<ILogger<PytorchModelService>>();
    var env = sp.GetRequiredService<IWebHostEnvironment>();
    
    try
    {
        return new PytorchModelService(logger, env);
    }
    catch (Exception ex)
    {
        logger.LogError(ex, "Failed to initialize PyTorch model. Continuing without PyTorch integration.");
        // Возвращаем null-object pattern или mock
        return new NullPytorchModelService();
    }
});

// Register PastureBiomassAnalyzer (now uses both PyTorch and GitHub Models API)
builder.Services.AddScoped<PastureBiomassAnalyzer>(sp =>
{
    var pytorchService = sp.GetService<IPytorchModelService>();
    return new PastureBiomassAnalyzer(pytorchService);
});
