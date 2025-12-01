namespace BioMasaWeb.Models;

public class BiomassAnalysisRequest
{
    public byte[] ImageData { get; set; } = Array.Empty<byte>();
    public string FileName { get; set; } = string.Empty;
    public DateTime MeasurementDate { get; set; } = DateTime.Now;
    public string State { get; set; } = string.Empty;
    public string DominantSpecies { get; set; } = string.Empty;
    public double NdviIndex { get; set; }
    public double AverageHeight { get; set; }
}

